''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import utils
from env import R2RBatch
from utils import padding_idx
from param import args
from collections import deque, defaultdict
from modeling import LXRTPretraining, Critic, Progressor
from tokenization import BertTokenizer
from optimization import BertAdam


scaler = torch.cuda.amp.GradScaler()


class BaseAgent(object):

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {} 
        self.losses = []
        
        self.logs = defaultdict(list)

    def write_results(self):
        output = [{'inst_idx': k, 'trajectory': v} for k, v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        self.loss = 0

        looped = False
        while True:
            for traj in self.rollout():
                if traj['inst_idx'] in self.results:
                    looped = True
                else:
                    self.results[traj['inst_idx']] = traj['path']
            if looped:
                break

    
class StopAgent(BaseAgent):  
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        # for t in range(30):  # 20 ep len + 10 (as in MP); planner paths
        for t in range(130):  # 120 ep len + 10 (emulating above); player paths
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else: 
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class Seq2SeqAgent(BaseAgent):

    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = {
      "left": ([0],[-1], [0]), # left
      "right": ([0], [1], [0]), # right
      "up": ([0], [0], [1]), # up
      "down": ([0], [0],[-1]), # down
      "forward": ([1], [0], [0]), # forward
      "<end>": ([0], [0], [0]), # <end>
      "<start>": ([0], [0], [0]), # <start>
      "<ignore>": ([0], [0], [0])  # <ignore>
    }

    env_actions_no_bracket = {
      "left": (0,-1, 0), # left
      "right": (0, 1, 0), # right
      "up": (0, 0, 1), # up
      "down": (0, 0,-1), # down
      "forward": (1, 0, 0), # forward
      "<end>": (0, 0, 0), # <end>
      "<start>": (0, 0, 0), # <start>
      "<ignore>": (0, 0, 0)  # <ignore>
    }
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, episode_len=20, path_type='planner_path', train_eval=None):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        # self.feature_size = 1024
        if args.feat_type == "resnet":
            self.feature_size = 2048
        elif args.feat_type == "resnet_clip":
            self.feature_size = 1024

        self.eval = train_eval


        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self.vln_bert = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=None,
            task_obj_predict=None,
            visual_losses='feat',
        ).cuda()

        t_total = 10000

        self.vln_bert_optimizer = torch.optim.AdamW(self.vln_bert.parameters(), lr=0.00001)
        
        self.critic = Critic(args.dropout).cuda()
        self.progressor = Progressor(768).cuda()
        self.progressor_optimizer = torch.optim.AdamW(self.progressor.parameters(), lr=0.00001)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=0.00001)
       
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.criterion_bt = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='none')


    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def convert_example_to_features(self, obs, max_seq_length=40):

        input_ids_list_batch = []
        input_mask_list_batch = []
        segment_ids_list_batch = []
        # lm_label_ids_list = []

        max_turns_len = 0
        last_id = []

        cur_l = []

        for ob in obs:
            cur_l.append(len(ob['raw_instruction'])-1)
            if max_turns_len < len(ob['raw_instruction']):
                max_turns_len = len(ob['raw_instruction'])


        for ob in obs:

            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []

            for raw_instruction in ob['raw_instruction']:
                tokens = self.tokenizer.tokenize(raw_instruction.strip())

                if len(tokens) > max_seq_length - 2:
                    tokens = tokens[-(max_seq_length - 2):]

                
                cur_id = 0
                # cur_id = 0
                segment_ids = [0]
                for token in tokens:
                    segment_ids.append(cur_id)


                last_id.append(segment_ids[-1])
                segment_ids.append(0)
                

                # concatenate lm labels and account for CLS, SEP, SEP
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # Mask & Segment Word
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)

            for _ in range(max_turns_len-len(input_ids_list)):

                input_ids_list.append([0]*max_seq_length)
                input_mask_list.append([0]*max_seq_length)
                segment_ids_list.append([0]*max_seq_length)


            input_ids_list_batch.append(input_ids_list)
            input_mask_list_batch.append(input_mask_list)
            segment_ids_list_batch.append(segment_ids_list)

        input_ids_tensor = torch.tensor(input_ids_list_batch, dtype=torch.long).cuda()
        input_mask_tensor = torch.tensor(input_mask_list_batch, dtype=torch.long).cuda()
        segment_ids_tensor = torch.tensor(segment_ids_list_batch, dtype=torch.long).cuda()


        return input_ids_tensor, input_mask_tensor, segment_ids_tensor, cur_l

    def _sort_batch(self, obs):
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != padding_idx)

        token_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)

        action_prompt = ""
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def get_input_feat_stack(self, t, f_t_stack, f_t, stack_mask, stack_len):

        max_stack_len = max(stack_len)
        assert max_stack_len == f_t_stack.size(1)+1 or max_stack_len == f_t_stack.size(1) or max_stack_len == f_t_stack.size(1)-1, "%d, %d"%(max_stack_len, f_t_stack.size(1))
        
        if max_stack_len > f_t_stack.size(1):
            feat_adding = torch.zeros(f_t_stack.size(0), 1, f_t_stack.size(-1)).cuda()
            mask_adding = torch.zeros(f_t_stack.size(0), 1).cuda()
            f_t_stack = torch.cat([f_t_stack, feat_adding], dim=1)
            stack_mask = torch.cat([stack_mask, mask_adding], dim=1)

        for i, sl in enumerate(stack_len):
            stack_mask[i,sl:] = 0
            f_t_stack[i,sl:] = 0
            stack_mask[i,sl:] = 0
            stack_mask[i,:sl] = 1
            f_t_stack[i,sl-1] = f_t[i]

        f_t_stack = f_t_stack[:, :max_stack_len]
        stack_mask = stack_mask[:, :max_stack_len]

        return f_t_stack, stack_mask

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, t, steps, obs, ended, logit, action_logit):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        next_turn = np.zeros(len(obs), dtype=np.int64)
        bt_label = np.ones(len(obs), dtype=np.int64)
        new_steps = [0] * len(obs)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
                bt_label[i] = args.ignoreid
            else:
                if ob['viewpoint'] == ob['path'][-1]:
                    a[i] = len(ob['candidate'])
                    assert ob['teacher'] == ob['viewpoint']
                elif self.feedback == 'teacher':
                    step = ob['path'][steps[i]:].index(ob['viewpoint'])
                    new_steps[i] = steps[i] + step
                    if t in ob["next_turn"]:
                        next_turn[i] = 1
                    for k, candidate in enumerate(ob['candidate']):
                        # if candidate['viewpointId'] == ob['path'][new_steps[i] + 1]:
                        if candidate['viewpointId'] == ob['path'][t+1]:
                            a[i] = k
                            break
                    else:
                        assert False
                else:
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['teacher']:  # Next view point
                            a[i] = k
                            break
                    else:  # Stop here
                        assert ob['teacher'] == ob['viewpoint'] 
                        a[i] = len(ob['candidate'])

        return torch.from_numpy(a).cuda(), logit, action_logit, torch.from_numpy(bt_label).cuda(), torch.from_numpy(next_turn).cuda(), new_steps

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                if args.server == "server":
                    self.env.env.sims[idx].makeAction(name, 0, 0)
                else:
                    self.env.env.sims[idx].makeAction([name], [0], [0])
            else:                       # Adjust
                if args.server == "server":
                    self.env.env.sims[idx].makeAction(*self.env_actions_no_bracket[name])
                else:
                    self.env.env.sims[idx].makeAction(*self.env_actions[name])

            if args.server == "server":
                state = self.env.env.sims[idx].getState()
            else:
                state = self.env.env.sims[idx].getState()[0]
            # if traj is not None:
            if traj is not None and state.location.viewpointId != traj[i]['path'][-1][0]:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                if args.server == "server":
                    while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                        take_action(i, idx, 'right')
                    assert select_candidate['viewpointId'] == \
                           self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                else:
                    while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:    # Turn right until the target
                        take_action(i, idx, 'right')
                    assert select_candidate['viewpointId'] == \
                           self.env.env.sims[idx].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])


    def get_scan_id(self, obs):
        scan_id = np.zeros((len(obs)), dtype=np.int64)
        for i,ob in enumerate(obs):
            scan_id[i] = self.scan_dict[ob['scan']]

        return torch.from_numpy(scan_id).cuda()

    def get_data_id(self, obs):
        data_id = np.zeros((len(obs)), dtype=np.int64)
        for i,ob in enumerate(obs):
            if ob['which_data'] == 'cvdn':
                data_id[i] = 0
            else:
                data_id[i] = 1

        return torch.from_numpy(data_id).cuda()

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        obs = np.array(self.env.reset())
        batch_size = len(obs)
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        perm_idx = list(range(batch_size))
        perm_obs = obs

        if self.vln_bert.training:
            data_id = self.get_data_id(perm_obs)


        # Record starting point
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            if train_rl:
                last_ndtw[i] = self.eval.ndtw(ob['scan'],traj[i]['path'], ob['path'])

        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), 
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        visited = [set() for _ in perm_obs]
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []

        
        self.loss_act = 0
        self.epi_len = 0
        ml_loss = 0.
        env_action = [None] * batch_size


        f_ts = deque()
        max_seq_length = 30
        input_ids, seq_mask, segment_ids, cur_l = self.convert_example_to_features(perm_obs, max_seq_length=max_seq_length)


        input_ids_len = input_ids.size(1)
        input_ids = input_ids.view(batch_size*input_ids_len, -1)
        segment_ids = segment_ids.view(batch_size*input_ids_len, -1)
        seq_mask = seq_mask.view(batch_size*input_ids_len, -1)
        h_l, lang_feat = self.vln_bert(
            'lang', input_ids, segment_ids, seq_mask, None,
            None, None, feats_len=None, candis_len=None
        )

        input_ids = input_ids.view(batch_size, input_ids_len, -1)
        h_l = h_l.view(batch_size, input_ids_len, -1)
        lang_feat = lang_feat.view(batch_size, input_ids_len, max_seq_length, -1)
        segment_ids = segment_ids.view(batch_size, input_ids_len, -1)
        seq_mask = seq_mask.view(batch_size, input_ids_len, -1)

        f_t_stack = torch.zeros(batch_size, 1, self.feature_size + args.angle_feat_size).cuda()
        stack_mask = torch.ones(batch_size, 1).cuda()
        stack_len = [1 for _ in range(batch_size)]
        h_t = torch.zeros(batch_size, self.feature_size + args.angle_feat_size).cuda()

        cur_p = [0] * batch_size
        h_l_p = h_l[:,0]

        steps = [0] * batch_size
        for t in range(self.episode_len):
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
                

            lang_feat_p = torch.cat([h_l_p.unsqueeze(1) ,lang_feat[range(batch_size), torch.tensor(cur_p)][:,1:]], dim=1)
            

            candis_len = candidate_feat.size(1)

            if t > 0:
                f_t_stack, stack_mask = self.get_input_feat_stack(t, f_t_stack, f_t, stack_mask, stack_len)

            f_ts_input_candi = torch.cat((f_t_stack, h_t.unsqueeze(1), candidate_feat), dim=1)
            visual_temp_mask = torch.cat([stack_mask, torch.ones(batch_size, 1).cuda()], dim=1).long()
            cand_visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long().cuda()

            visual_candi_temp_mask = torch.cat((visual_temp_mask, cand_visual_temp_mask), dim=-1)



            h_t, h_l_p, logit, action_logit, btrack_score, house_logit, lr_h = self.vln_bert(
            'vis', lang_feat_p, segment_ids[range(batch_size), torch.tensor(cur_p)], seq_mask[range(batch_size), torch.tensor(cur_p)], None,
            f_ts_input_candi, visual_candi_temp_mask, h_t=h_t, feats_len=stack_len, prox_pos=f_t_stack.size(1)+1, candis_len=candis_len)
            _, next_turn_logit = self.progressor(h_t, h_l, cur_p, cur_l)


            
            candidate_mask = utils.length2mask(candidate_leng)


            logit.masked_fill_(candidate_mask, -float('inf'))

            hidden_states.append(lr_h)

            if 'test' in self.env.splits:
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1

            
            # Supervised training
            if 'test' not in self.env.splits:
                target, logit, action_logit, bt_label, next_turn_label, steps = self._teacher_action(t, steps, perm_obs, ended, logit, action_logit)

                
               
                for i in range(len(cur_p)):
                    if cur_p[i] == cur_l[i]:
                        continue

                    if self.feedback == 'teacher':
                        cur_p[i] += next_turn_label[i].item()
                    else:
                        cur_p[i] += torch.argmax(next_turn_logit[i]).item()

                tmp_loss_bt = 0
                tmp_loss = self.criterion(logit, target)
                if not train_rl:
                    tmp_loss += self.criterion(next_turn_logit.view(-1, 2), next_turn_label.view(-1))

                if not math.isinf(tmp_loss):
                    ml_loss += tmp_loss


            if self.feedback == 'teacher':
                a_t = target                
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)  
                a_t = a_t
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                             
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)

                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())     
                entropys.append(c.entropy())                               
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            cpu_a_t = a_t.cpu().numpy()

            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    
                    cpu_a_t[i] = -1            

                stack_len[i] += 1


            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)

            obs = np.array(self.env._get_obs())

            perm_obs = obs[perm_idx]                  

            


            if train_rl:
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    if self.eval is not None:
                        ndtw_score[i] = self.eval.ndtw(ob['scan'],traj[i]['path'], ob['path'])

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:                              
                            if dist[i] < 3.0:                             
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:                                         
                                reward[i] = -2.0
                        else:                                            
                            reward[i] = ndtw_score[i] - last_ndtw[i]

                            if reward[i] > 0.0:                        
                                reward[i] = 1.0 
                            elif reward[i] <= 0.0:
                                reward[i] = -1.0 
                            else:
                                raise NameError("The action doesn't change the move")

                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score


            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            if ended.all():
                break


        
        if train_rl:

            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
                
            lang_feat_p = torch.cat([h_l_p.unsqueeze(1) ,lang_feat[range(batch_size), torch.tensor(cur_p)][:,1:]], dim=1)
            
            candis_len = candidate_feat.size(1)

            if t > 0:
                f_t_stack, stack_mask = self.get_input_feat_stack(t, f_t_stack, f_t, stack_mask, stack_len)
            f_ts_input_candi = torch.cat((f_t_stack, h_t.unsqueeze(1), candidate_feat), dim=1)
            visual_temp_mask = torch.cat([stack_mask, torch.ones(batch_size, 1).cuda()], dim=1).long()
            cand_visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long().cuda()

            visual_candi_temp_mask = torch.cat((visual_temp_mask, cand_visual_temp_mask), dim=-1)
            

            _, _, _, _, _, _, last_h_= self.vln_bert(
            'vis', lang_feat_p, segment_ids[range(batch_size), torch.tensor(cur_p)], seq_mask[range(batch_size), torch.tensor(cur_p)], None,
            f_ts_input_candi, visual_candi_temp_mask, h_t=h_t, feats_len=stack_len, prox_pos=f_t_stack.size(1)+1, candis_len=candis_len)

            rl_loss = 0.


            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * 0.9 + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            rl_loss /= total


            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if self.epi_len == 0:
            self.epi_len = self.episode_len

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.epi_len)   

        return traj


    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback

        self.losses = []
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
            self.progressor.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
            self.progressor.eval()
        super(Seq2SeqAgent, self).test()

    def train(self, n_iters, isFinetune = False, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.vln_bert.train()
        self.critic.train()
        self.progressor.train()
        self.losses = []
        for i in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            if isFinetune:
                self.critic_optimizer.zero_grad()
            self.progressor_optimizer.zero_grad()
            self.loss = 0
            if isFinetune:
                self.feedback = 'teacher'
                
                self.rollout(train_ml=0.20, train_rl=False)

                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True)
            else:
                self.feedback = 'teacher'
                self.rollout(train_ml=1.0, train_rl=False)
  
            self.loss.backward()
            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)
            if isFinetune:
                torch.nn.utils.clip_grad_norm(self.critic.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.progressor.parameters(), 40.)
            self.vln_bert_optimizer.step()
            if isFinetune:
                self.critic_optimizer.step()
            self.progressor_optimizer.step()

    def save(self, encoder_path):
        ''' Snapshot models '''
        torch.save(self.vln_bert.state_dict(), encoder_path)
        # torch.save(self.decoder.state_dict(), decoder_path)

    def save_progressor(self, encoder_path):
        ''' Snapshot models '''
        torch.save(self.progressor.state_dict(), encoder_path)

    def save_critic(self, encoder_path):
        ''' Snapshot models '''
        torch.save(self.critic.state_dict(), encoder_path)

    def save_bert(self, encoder_path):
        ''' Snapshot models '''
        torch.save(self.vln_bert.state_dict(), encoder_path)

    def load(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load(path)

        self.vln_bert.load_state_dict(state_dict, strict=False)

