''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
from param import args
import torch


def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

def length2mask_1(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 2).unsqueeze(1)).cuda()
    return mask

def ndtw_initialize():
    ndtw_criterion = {}
    scan_gts_dir = 'data/id_paths.json'
    with open(scan_gts_dir) as f_:
        scan_gts = json.load(f_)
    all_scan_ids = []
    for key in scan_gts:
        path_scan_id = scan_gts[key][0]
        # print('path_scan_id', path_scan_id)
        if path_scan_id not in all_scan_ids:
            all_scan_ids.append(path_scan_id)
            ndtw_graph = ndtw_graphload(path_scan_id)
            ndtw_criterion[path_scan_id] = DTW(ndtw_graph)
    return ndtw_criterion

def pad_instr_tokens(instr_tokens, maxlength=20):


    if len(instr_tokens) > maxlength - 2: # -2 for [CLS] and [SEP]
        instr_tokens = instr_tokens[:(maxlength-2)]

    instr_tokens = ['[CLS]'] + instr_tokens + ['[SEP]']
    num_words = len(instr_tokens)  # - 1  # include [SEP]
    instr_tokens += ['[PAD]'] * (maxlength-len(instr_tokens))

    assert len(instr_tokens) == maxlength

    return instr_tokens, num_words

def new_simulator():
    import MatterSim
    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    if args.server == "server":
        sim.init()
    else:
        sim.initialize()

    return sim

def angle_feature(heading, elevation):
    import math
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (args.angle_feat_size // 4),
                    dtype=np.float32)

def angle_pos_feature(heading, elevation, x, y, z):
    import math
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation), x, y, z] * (args.angle_feat_size // 4),
                    dtype=np.float32)

def get_point_angle_feature(baseViewId=0):
    sim = new_simulator()

    feature = np.empty((36, args.angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            if args.server == "server":
                sim.newEpisode('ZMojNkEp431', '2f4d90acd4024c269fb0efe49a8ac540', 0, math.radians(-30))
            else:
                sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            if args.server == "server":
                sim.makeAction(0, 1.0, 1.0)
            else:
                sim.makeAction([0], [1.0], [1.0])
        else:
            if args.server == "server":
                sim.makeAction(0, 1.0, 0)
            else:
                sim.makeAction([0], [1.0], [0])
        if args.server == "server":
            state = sim.getState()
        else:
            state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_feature(heading, state.elevation)
    return feature

def get_point_angle_pos_feature(baseViewId=0):
    sim = new_simulator()

    feature = np.empty((36, args.angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            if args.server == "server":
                sim.newEpisode('ZMojNkEp431', '2f4d90acd4024c269fb0efe49a8ac540', 0, math.radians(-30))
            else:
                sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            if args.server == "server":
                sim.makeAction(0, 1.0, 1.0)
            else:
                sim.makeAction([0], [1.0], [1.0])
        else:
            if args.server == "server":
                sim.makeAction(0, 1.0, 0)
            else:
                sim.makeAction([0], [1.0], [0])

        if args.server == "server":
            state = sim.getState()
        else:
            state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_pos_feature(heading, state.elevation, state.location.x, state.location.y, state.location.z)
    return feature

def get_all_point_angle_feature():
    return [get_point_angle_feature(baseViewId) for baseViewId in range(36)]
    # return [get_point_angle_pos_feature(baseViewId) for baseViewId in range(36)]

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<NAV>', '<ORA>', '<TAR>']
padding_idx = base_vocab.index('<PAD>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3], 
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits):
    data = []
    for split in splits:
        # assert split in ['train', 'R2R_train', 'val_seen', 'val_unseen', 'test']
        with open('tasks/NDH_full/data/%s.json' % split) as f:
            data += json.load(f)
    return data


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character
  
    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
                
        self.add_word('<BOS>')

    def vocab_size(self):
        return len(self.index_to_word)

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentences, seps=None):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        if type(sentences) is not list:
            sentences = [sentences]
            seps = [seps]
        for sentence, sep in zip(sentences, seps):
            if sep is not None:
                encoding.append(self.word_to_index[sep])
            for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                if word in self.word_to_index:
                    encoding.append(self.word_to_index[word])
                else:
                    encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return np.array(encoding[prefix_cut:])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1]) # unreverse before output

    def encode_dial(self, sentences, seps=None):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        if type(sentences) is list:
            for sentence, sep in zip(sentences, seps):
                if sep is not None:
                    encoding.append(self.word_to_index[sep])
                for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                    if word in self.word_to_index:
                        encoding.append(self.word_to_index[word])
                    else:
                        encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return encoding[prefix_cut:]


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for turn in item['dialog_history']:
            count.update(t.split_sentence(turn['message']))
    vocab = list(start_vocab)

    # Add words that are object targets.
    targets = set()
    for item in data:
        target = item['target']
        targets.add(target)
    vocab.extend(list(targets))

    # Add words above min_count threshold.
    for word, num in count.most_common():
        if word in vocab:  # targets strings may also appear as regular vocabulary.
            continue
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

