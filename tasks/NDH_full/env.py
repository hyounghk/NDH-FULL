''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
import utils
from random import randint
from param import args

from utils import load_datasets, load_nav_graphs, pad_instr_tokens

csv.field_size_limit(sys.maxsize)



class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100, blind=False):
        if feature_store:
            print('Loading image features from %s' % feature_store)
            if blind:
                print("... and zeroing them out for 'blind' evaluation")
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            self.features = {}
            with open(feature_store, "r") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    self.image_h = int(item['image_h'])
                    self.image_w = int(item['image_w'])
                    self.vfov = int(item['vfov'])
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    if not blind:
                        if args.feat_type == "resnet":
                            self.features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                dtype=np.float32).reshape((36, 2048))
                        elif args.feat_type == "resnet_clip":
                            self.features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                dtype=np.float32).reshape((36, 1024))
                    else:
                        if args.feat_type == "resnet":
                            self.features[long_id] = np.zeros((36, 2048), dtype=np.float32)
                        elif args.feat_type == "resnet_clip":
                            self.features[long_id] = np.zeros((36, 1024), dtype=np.float32)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.batch_size = batch_size
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            if args.server == "server":
                sim.init()
            else:
                sim.initialize()

            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            if args.server == "server":
                self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
            else:
                self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])
  
    def getStates(self):
        feature_states = []
        for i, sim in enumerate(self.sims):
            if args.server == "server":
                state = sim.getState()
            else:
                state = sim.getState()[0]
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states


    def makeActions(self, actions):

        for i, (index, heading, elevation) in enumerate(actions):
            if args.server == "server":
                self.sims[i].makeAction(index, heading, elevation)
            else:
                self.sims[i].makeAction([index], [heading], [elevation])


class R2RBatch():

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 path_type='player_path_total', blind=False, evaluator=None):
        self.buffered_state_dict = {}
        self.sim = utils.new_simulator()
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size, blind=blind)
        self.data = []
        self.scans = []
        self.splits = splits
        for item in load_datasets(splits):
            if item.get('dialog_history') is not None:

                next_turn = []
                for it in item["dialog_history_list"]:
                    if len(it) == 0:
                        continue
                    next_turn.append(it[-1]["nav_idx"])

                
                target = []
                target.append(item['target'])
                tar_seps = []
                tar_seps.append('<TAR>')

                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['inst_idx'] = item['inst_idx']
                new_item['next_turn'] = next_turn
                    
                sentences = []
                for i, dialog in enumerate(item['dialog_history_list']):
                    sentence = ''
                    sentence += item['target']
                    for j, turn in enumerate(dialog):
                        sentence += " [SEP] " + turn["message"]

                    sentences.append(sentence)

                new_item['instructions'] = sentences
                new_item['raw_instruction'] = sentences
                new_item['which_data'] = "cvdn"
                new_item['instr_encoding'] = sentences

                if path_type == 'trusted_path' and 'test' not in splits:

                    planner_goal = item['player_path_total'][-1]  
                    if planner_goal in item['player_path'][1:]:  
                        new_item['trusted_path'] = item['player_path_total'] 
                    else:
                        new_item['trusted_path'] = item['player_path_total']

                self.data.append(new_item)

            else:
                for j,instr in enumerate(item['instructions'][:1]):
                    if item['scan'] not in self.env.featurized_scans: 
                        continue
                    new_item = dict(item)
                    new_item['inst_idx'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    new_item['raw_instruction'] = instr
                    new_item['trusted_path'] = item['path'][:]
                    new_item['which_data'] = "r2r"
                    self.data.append(new_item)
                    self.scans.append(item['scan'])

        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self.path_type = path_type
        self.angle_feature = utils.get_all_point_angle_feature()
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _load_nav_graphs(self):
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):

        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        cur_elevatoin = viewId // 12
        cur_heading = viewId % 12
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    if args.server == "server":
                        self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                    else:
                        self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    if args.server == "server":
                        self.sim.makeAction(0, 1.0, 1.0)
                    else:
                        self.sim.makeAction([0], [1.0], [1.0])
                else:
                    if args.server == "server":
                        self.sim.makeAction(0, 1.0, 0)
                    else:
                        self.sim.makeAction([0], [1.0], [0])

                if args.server == "server":
                    state = self.sim.getState()
                else:
                    state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                
                next_elevatoin = ix // 12
                next_heading = ix % 12


             
                for j, loc in enumerate(state.navigableLocations[1:]):
                    distance = _loc_distance(loc)

                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)

                    if (loc.viewpointId not in adj_dict or distance < adj_dict[loc.viewpointId]['distance']):

                        temp = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'angle_feat': angle_feat,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
                        
                        adj_dict[loc.viewpointId] = temp

            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'angle_feat']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:

                c_new = c.copy()
                ix = c_new['pointId']

                next_elevatoin = ix // 12
                next_heading = ix % 12

                
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                # angle_feat = c_new['angle_feat']
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')

                candidate_new.append(c_new)

            return candidate_new

    def _get_obs(self, previous_loc=None):
        obs = []
        for i,(feature,state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            obs.append({
                'inst_idx': item['inst_idx'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature[state.viewIndex,:],
                "candidate": candidate,
                'step': state.step,
                'navigableLocations': state.navigableLocations,
                'instructions': item['instructions'],
                'teacher': self._shortest_path_action(state, item[self.path_type][-1]) if 'test' not in self.splits else None,
                'path': item['trusted_path']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            obs[-1]['raw_instruction'] = item['raw_instruction']
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['end_panos'][0]]
            obs[-1]['which_data'] = item['which_data']
            obs[-1]['next_turn'] = item['next_turn']
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        # if 'train' in self.splits:
        if 'test' not in self.splits:
            viewpointIds = [item[self.path_type][0] for item in self.batch]
        else:
            viewpointIds = [item['start_pano_list'][0]['pano'] for item in self.batch]

        headings = []
        for item in self.batch:
            if item['which_data'] == 'cvdn': 
                headings.append(item['start_pano_list'][0]['heading'])
            else:
                headings.append(item['heading'])

        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()   

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()


