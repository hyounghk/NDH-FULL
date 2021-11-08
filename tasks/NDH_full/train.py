import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args


import torch.multiprocessing as mp
import torch.distributed as dist

import pprint
pp = pprint.PrettyPrinter(indent=4)

import warnings
warnings.filterwarnings("ignore")

prefix=args.prefix

RESULT_DIR = 'tasks/NDH_full/results/' + prefix + '/'
SNAPSHOT_DIR = 'tasks/NDH_full/snapshots/' + prefix + '/'
PLOT_DIR = 'tasks/NDH_full/plots/' + prefix + '/'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
IMAGENET_FEATURES_CLIP = 'img_features/CLIP-ResNet-50-views.tsv'

agent_type = 'seq2seq'

if args.feat_type == "resnet":
    features = IMAGENET_FEATURES
elif args.feat_type == "resnet_clip":
    features = IMAGENET_FEATURES_CLIP

batch_size = args.batch_size
# word_embedding_size = 256
word_embedding_size = 256
action_embedding_size = 32
target_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
learning_rate = 0.0001
weight_decay = 0.0005


def train(train_env, train_eval, n_iters, path_type, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix,
    log_every=100, val_envs=None):
    if val_envs is None:
        val_envs = {}

    print("reinforce: ", args.reinforce)
    isFinetune = args.reinforce
    if agent_type == 'seq2seq':
        agent = Seq2SeqAgent(train_env, "", max_episode_len, train_eval=train_eval)
    else:
        sys.exit("Unrecognized agent_type '%s'" % agent_type)
    print('Training a %s agent with %s feedback' % (agent_type, feedback_method))

    data_log = defaultdict(list)
    start = time.time()
    print('Start training')
    best_reduction = 0.0
    best_ndtw = 0.0
    best_sr = 0.0
    best_iter = 0
    best_iter_ndtw = 0
    best_iter_sr = 0
    for idx in range(0, n_iters, log_every):

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        agent.train(interval, feedback=feedback_method, isFinetune=isFinetune)
        train_losses = np.array(agent.losses)
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            agent.test(use_dropout=False, feedback='argmax')
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
             data_log['%s %s' % (env_name, metric)].append(val)
             if metric in ['success_rate', 'oracle success_rate', 'nav_error', 'spl', 'suc_length', 'final_length', 'length', 'oracle path_success_rate', 'dist_to_end_reduction', 'ndtw']:
                 loss_str += ', %s: %.3f' % (metric, val)

                 if env_name == "val_unseen_seq" and metric == 'success_rate' and val > best_sr:
                    best_sr = val
                    best_iter_sr = iter

                 if env_name == "val_unseen_seq" and metric == 'dist_to_end_reduction' and val > best_reduction:
                    best_reduction = val
                    best_iter = iter

                 if env_name == "val_unseen_seq" and metric == 'ndtw' and val > best_ndtw:
                    best_ndtw = val
                    best_iter_ndtw = iter
                    
                    split_string = "-".join(train_env.splits)

                    critic_path = '%s%s_%s_critic_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
                    bert_path = '%s%s_%s_bert_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
                    prog_path = '%s%s_%s_prog_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)

                    agent.save_critic(critic_path)
                    agent.save_bert(bert_path)
                    agent.save_progressor(prog_path)

        if env_name == "val_unseen_seq":
            loss_str += ', %s: %.3f at %d' % ('BEST', best_reduction, best_iter)
            loss_str += ', %s: %.3f at %d' % ('BEST_SR', best_sr, best_iter_sr)
            loss_str += ', %s: %.3f at %d' % ('BEST_NDTW', best_ndtw, best_iter_ndtw)
        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                                         iter, float(iter)/n_iters*100, loss_str))
        agent.env = train_env      
        
    print('Finish training')
def setup():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def train_val(path_type, max_episode_len, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind):
  
    setup()

    train_env = R2RBatch(features, batch_size=batch_size, splits=['train_seq'], tokenizer=None,
                         path_type=path_type, blind=blind)
    train_eval = Evaluation(['train_seq'], path_type=path_type)

    val_envs = {}
    for split in ['test_seq', 'val_unseen_seq']:
        evalu = Evaluation([split], path_type=path_type)
        r2rbatch = R2RBatch(features, batch_size=batch_size, splits=[split], 
                tokenizer=None, path_type=path_type, blind=blind, evaluator=evalu)
        val_envs[split] = (r2rbatch, evalu)

    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size

    train(train_env, train_eval, n_iters,
          path_type, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix, val_envs=val_envs)

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
if __name__ == "__main__":
    
    assert args.path_type in ['planner_path', 'player_path', 'trusted_path']
    assert args.feedback in ['sample', 'teacher', 'argmax']
    assert args.eval_type in ['val', 'test']

    blind = args.blind
    path_type = args.path_type

    if path_type == 'planner_path':
        max_episode_len = 20  
    else:
        max_episode_len = 40  

    MAX_INPUT_LENGTH = 120 * 6  

    feedback_method = args.feedback
    n_iters = 20000

    model_prefix = '%s-seq2seq-%s-%d-%s-imagenet' % (
    args.eval_type, path_type, max_episode_len, feedback_method)
    if blind:
        model_prefix += '-blind'

    if args.eval_type == 'val':
        train_val(path_type, max_episode_len, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)
    else:
        train_test(path_type, max_episode_len, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)



