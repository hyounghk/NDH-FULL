import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from param import args

from file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except Importtokenization:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualConfig(object):
    # VISUAL_LOSSES = ['obj', 'attr', 'feat']
    VISUAL_LOSSES = ['feat']
    def __init__(self,
                 l_layers=9,
                 x_layers=5,
                 r_layers=5):
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers

        if args.feat_type == "resnet":
            self.visual_feat_dim = 2048+128
        elif args.feat_type == "resnet_clip":
            self.visual_feat_dim = 1024+128

        self.visual_pos_dim = 4

        self.obj_id_num = 1600
        self.attr_id_num = 400

        self.visual_losses = self.VISUAL_LOSSES

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


VISUAL_CONFIG = VisualConfig()


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)
        self.token_area_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_area_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, attention_gate=None, is_lang_att=None, vis_position=None, is_visual_att=False, feat_len=None, prox_pos=None, candis_len=None, area_id=None):

        if is_visual_att and True:
            positioned_context = context
            for i in range(hidden_states.size(0)):
                hidden_states[i, :feat_len[i], :] = hidden_states[i, :feat_len[i], :] + vis_position[i, :feat_len[i], :]
                positioned_context[i, :feat_len[i], :] = positioned_context[i, :feat_len[i], :] + vis_position[i, :feat_len[i], :]
        else:
            positioned_context = context

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(positioned_context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            if attention_gate is not None:
                attention_scores = attention_scores * attention_gate + attention_mask
            else:
                attention_scores = attention_scores + attention_mask

        if is_visual_att and True:

            attention_scores[:,:,prox_pos:, prox_pos:] = -100000.0

            for i in range(prox_pos-1):
                attention_scores[:,:,i, i+1:] = -100000.0

                if i-1 >= 0:
                    attention_scores[:,:,i, :i-1] = -100000.0

            for k in range(hidden_states.size(0)):
                attention_scores[k, :, prox_pos:, :feat_len[k]-1] = -100000.0

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, ctx_att_gate=None, area_id=None):
        output, _ = self.att(input_tensor, ctx_tensor, ctx_att_mask, ctx_att_gate, area_id=area_id)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask, attention_gate=None, is_lang_att=None, vis_position=None, is_visual_att=False, feat_len=None, prox_pos=None, candis_len=None, area_id=None):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output, self_score = self.self(input_tensor, input_tensor, attention_mask, attention_gate, is_lang_att, vis_position, is_visual_att, feat_len, prox_pos, candis_len, area_id=area_id)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, self_score


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, is_lang_att=None, vis_position=None, is_visual_att=None, feat_len=None, prox_pos=None):
        attention_output, _ = self.attention(hidden_states, attention_mask, is_lang_att=is_lang_att, vis_position=vis_position, is_visual_att=is_visual_att, feat_len=feat_len, prox_pos=prox_pos)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads




    def cross_att(self, lang_input, lang_attention_mask, lang_attention_gate, visn_input, visn_attention_mask, visn_attention_gate, area_id=None):
        # Cross Attention

        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask, ctx_att_gate=visn_attention_gate, area_id=area_id)

        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask, ctx_att_gate=lang_attention_gate, area_id=area_id)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, lang_attention_gate, visn_input, visn_attention_mask, visn_attention_gate, vis_position_embeddings=None, feat_len=None, prox_pos=None, candis_len=None, area_id=None):
        # Self Attention
        lang_att_output, lang_att_score = self.lang_self_att(lang_input, lang_attention_mask, lang_attention_gate, is_lang_att=True, candis_len=candis_len, area_id=area_id)
        visn_att_output, visn_att_score = self.visn_self_att(visn_input, visn_attention_mask, visn_attention_gate, vis_position=vis_position_embeddings, is_visual_att=True, feat_len=feat_len, prox_pos=prox_pos, area_id=area_id)
        return lang_att_output, visn_att_output, lang_att_score, visn_att_score

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)



    def forward(self, layer_num, lang_feats, lang_attention_mask, lang_attention_mask_cross, visn_attention_mask_cross, lang_attention_gate,
                      visn_feats, visn_attention_mask, visn_attention_gate, vis_position_embeddings=None, feat_len=None, prox_pos=None, area_id=None, candis_len=None):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_attention_gate = None

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask, lang_attention_gate, visn_att_output, visn_attention_mask, None, area_id)

        lang_att_output, visn_att_output, lang_att_score, visn_att_score = self.self_att(lang_att_output, lang_attention_mask, lang_attention_gate, visn_att_output, visn_attention_mask, None, vis_position_embeddings, feat_len, prox_pos, candis_len, area_id)

        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output, lang_att_score, visn_att_score, lang_attention_mask_cross, visn_attention_mask_cross, lang_attention_mask, visn_attention_mask, lang_attention_gate


class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim

        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        output = x

        output = self.dropout(output)
        return output


class VisualFeatEncoder_candi(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim

        self.visn_fc = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        output = x

        output = self.dropout(output)
        return output


class VisualFeatEncoder_candi_angle(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim

        self.visn_fc = nn.Linear(128, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        output = x

        output = self.dropout(output)
        return output

class multihead_merge_feat(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        self.visn_fc = nn.Linear(config.hidden_size*3, 1)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        output = x
        return output


class score_BT(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim

        self.visn_fc = nn.Linear(config.hidden_size*2, 1)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        output = x

        return output

class multihead_merge(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        self.visn_fc = nn.Linear(12, 1)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        output = x
        return output


class LXRTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.visn_fc = VisualFeatEncoder(config)
        self.visn_fc_candi = VisualFeatEncoder_candi(config)
        self.visn_candi_angle = VisualFeatEncoder_candi_angle(config)
        self.lang_fc_candi = VisualFeatEncoder_candi(config)
        self.multihead_merge = multihead_merge(config)
        self.multihead_merge_feat = multihead_merge_feat(config)
        self.score_BT = score_BT(config)
        self.vis_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)

        # Number of layers
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
              (self.num_l_layers, self.num_x_layers, self.num_r_layers))

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, mode, lang_feats, lang_attention_mask,
                lang_attention_gate=None, visn_feats=None, visn_attention_mask=None, visn_attention_gate=None, h_t=None,feats_len=None, prox_pos=None, candis_len=None, area_id=None):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        

        # Run language layers
        if mode == 'lang':
            # with torch.no_grad():
            for layer_module in self.layer:
                lang_feats = layer_module(lang_feats, lang_attention_mask)

            visual_attention_scores = None
            visual_attention_scores_backtrack = None
            visual_attention_scores_2 = None
            visn_feats_const = None
            forback_logits = None
        else:
        # Run relational layers

            batch_size = lang_feats.size(0)
            
            angle_feat = visn_feats[:,prox_pos:,-128:]
            visn_feats = self.visn_fc(visn_feats)

            vis_position_ids = torch.zeros(visn_feats.size(0), visn_feats.size(1)).long().cuda(lang_feats.device)

            
            with torch.no_grad():
                vis_zeros = torch.zeros(visn_feats.size(0), prox_pos, visn_feats.size(-1)).cuda(lang_feats.device)
                lang_zeros = torch.zeros(visn_feats.size(0), 80, visn_feats.size(-1)).cuda(lang_feats.device)


            for i in range(visn_feats.size(0)):
                vis_position_ids[i,:feats_len[i]] = torch.arange(start=feats_len[i]-1, end=-1, step=-1, dtype=torch.long, device=visn_feats.device)

            vis_position_embeddings = self.vis_position_embeddings(vis_position_ids)


            lang_attention_mask_cross = None
            visn_attention_mask_cross = None
            for i, layer_module in enumerate(self.x_layers):
                lang_feats, visn_feats, lang_att_score, visn_att_score, lang_attention_mask_cross, visn_attention_mask_cross, lang_attention_mask, visn_attention_mask, lang_attention_gate = layer_module(i, lang_feats, lang_attention_mask, lang_attention_mask_cross, visn_attention_mask_cross, lang_attention_gate,
                    visn_feats, visn_attention_mask, visn_attention_gate, vis_position_embeddings, feat_len=feats_len, prox_pos=prox_pos, area_id=area_id, candis_len=candis_len)
                
                vis_lang_fuse = self.visn_fc_candi(torch.cat([visn_feats[:, prox_pos-1, :], lang_feats[:, 0, :], visn_feats[:, prox_pos-1, :]*lang_feats[:, 0, :]], dim=-1))
                visn_feats = torch.cat([visn_feats[:, :prox_pos-1, :], vis_lang_fuse.unsqueeze(1), visn_feats[:, prox_pos:, :]], dim=1)

            lang_feats_ext = lang_feats[:, 0:1, :].repeat(1, visn_feats.size(1), 1)
            visual_attention_scores_source_lang = self.multihead_merge_feat(torch.cat([visn_feats, lang_feats_ext, visn_feats*lang_feats_ext], dim=-1)).squeeze(-1)
            visual_attention_scores_source = self.multihead_merge(visn_att_score[:,:,prox_pos-1,:].permute(0,2,1)).squeeze(-1)
                
            visual_attention_scores_2 = visual_attention_scores_source_lang[:, prox_pos:prox_pos+candis_len]
            visual_attention_scores_1 = visual_attention_scores_source[:, prox_pos:prox_pos+candis_len]
            visual_attention_scores = visual_attention_scores_1

            visual_attention_scores_backtrack = torch.zeros(visn_feats.size(0), 1).cuda(lang_feats.device)

            
            forback_scores = []
            forback_logits = []
            for i in range(visn_feats.size(0)):

                if feats_len[i] > 1:
                    backward_score = visual_attention_scores_source_lang[i,feats_len[i]-2:feats_len[i]-1]
                else:
                    backward_score = torch.tensor(-float('inf')).unsqueeze(-1).cuda(lang_feats.device)

                forward_score = visual_attention_scores_source_lang[i,prox_pos-1:prox_pos]

                forback_logit = torch.cat([backward_score, forward_score], dim=-1)
                forback_score = torch.softmax(forback_logit, dim=-1)
                forback_logits.append(forback_logit)
                forback_scores.append(forback_score)

                if feats_len[i] > 1:
                    visual_attention_scores_backtrack[i,:] = visual_attention_scores_source[i,feats_len[i]-2]
                else:
                    visual_attention_scores_backtrack[i,:] = -float('inf')

            forback_scores = torch.stack(forback_scores).cuda(lang_feats.device)
            forback_logits = torch.stack(forback_logits).cuda(lang_feats.device)

        return lang_feats, visn_feats, visual_attention_scores, forback_logits, visual_attention_scores_backtrack


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class V_BertPooler(nn.Module):
    def __init__(self, config):
        super(V_BertPooler, self).__init__()
        if args.feat_type == "resnet":
            self.dense = nn.Linear(config.hidden_size, 2048+128)
        elif args.feat_type == "resnet_clip":
            self.dense = nn.Linear(config.hidden_size, 1024+128)

        self.activation = nn.Tanh()

    def forward(self, hidden_states, prox_pos):
        first_token_tensor = hidden_states[:, prox_pos-1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class VL_BertPooler(nn.Module):
    def __init__(self, config):
        super(VL_BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class GradReverse(torch.autograd.Function):
    scale = 1.3
    @staticmethod
    def forward(ctx, x):
        x.view_as(x)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return GradReverse.scale * result * grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class BertVisualAnswerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 2)
        )

    def forward(self, hidden_states, feat_len):
        hidden_states = grad_reverse(hidden_states[:, feat_len-1])
        return self.logit_fc(hidden_states)


class BertVisualObjHead(nn.Module):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        visual_losses = visual_losses.split(",")
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses
        self.decoder_dict = nn.ModuleDict({
            key: nn.Linear(config.hidden_size, VISUAL_CONFIG.visual_loss_config[key][0])
            for key in self.visual_losses
        })

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):

            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            print(archive_file)
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path == 'bert-base-uncased':
                try:
                    print("The BERT-weight-downloading query to AWS was time-out;" 
                          "trying to download from UNC servers")
                    archive_file = "https://nlp.cs.unc.edu/data/bert/bert-base-uncased.tar.gz"
                    resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
                except EnvironmentError:
                    print("The weight-downloading still crashed with link: %s, "
                          "please check your network connection" % archive_file)
                    return None
            else:
                logger.error(
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url but couldn't find any file "
                        "associated to this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                            archive_file))
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        # if len(missing_keys) > 0:
        #     logger.info("Weights of {} not initialized from pretrained model: {}".format(
        #         model.__class__.__name__, missing_keys))
        # if len(unexpected_keys) > 0:
        #     logger.info("Weights from pretrained model not used in {}: {}".format(
        #         model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class LXRTModel(BertPreTrainedModel):
    """LXRT Model."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config)
        self.pooler = BertPooler(config)
        # self.pooler_area = BertPooler_area(config)
        self.v_pooler = V_BertPooler(config)
        self.vl_pooler = VL_BertPooler(config)
        self.apply(self.init_bert_weights)
        self.house_head = BertVisualAnswerHead(config)

        self.action_embedding = nn.Embedding(61, 128)

        self.cand_act_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 128),
        )

        self.cand_act_to_feat = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 128),
        )
        self.logit_scale = nn.Parameter(torch.ones([]))

    def forward(self, mode, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None, h_t=None, feats_len=None, prox_pos=None, candis_len=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -100000.0

        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)

            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -100000.0
        else:
            extended_visual_attention_mask = None

        if mode == 'lang':
            embedding_output = self.embeddings(input_ids, token_type_ids)
            lang_feats, visn_feats, candi_score, action_score, _= self.encoder(
                mode,
                embedding_output,
                extended_attention_mask,
                lang_attention_gate=attention_mask,
                visn_feats=visual_feats,
                visn_attention_mask=extended_visual_attention_mask,
                visn_attention_gate=visual_attention_mask,
                h_t=h_t,
                feats_len=feats_len,
                prox_pos=prox_pos,
                candis_len=candis_len,
                area_id=token_type_ids)
            visn_feats_h_t = None
            house_logit = None
            lr_h = None
            candi_score_bt = None
            scors1, scors2 = None, None
            pooled_output = self.pooler(lang_feats)
        else:

            lang_feats, visn_feats, candi_score, action_score, candi_score_bt = self.encoder(
                mode,
                input_ids,
                extended_attention_mask,
                lang_attention_gate=attention_mask,
                visn_feats=visual_feats,
                visn_attention_mask=extended_visual_attention_mask,
                visn_attention_gate=visual_attention_mask,
                h_t=h_t,
                feats_len=feats_len,
                prox_pos=prox_pos,
                candis_len=candis_len,
                area_id=token_type_ids)
            visn_feats_h_t = self.v_pooler(visn_feats, prox_pos)
            house_logit = None
            lr_h = self.vl_pooler(visn_feats[:,prox_pos-1,:])

            pooled_output = self.pooler(lang_feats)

        


        return (lang_feats, visn_feats), pooled_output, candi_score, action_score, candi_score_bt, visn_feats_h_t, None, lr_h

class LXRTPretraining(BertPreTrainedModel):
    def __init__(self,
                 config,
                 task_mask_lm=True,
                 task_obj_predict=True,
                 visual_losses='feat'):
        super().__init__(config)
        # Configuration
        self.config = config
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict

        # LXRT backbone
        self.bert = LXRTModel(config)

    def forward(self, mode, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, visual_attention_mask=None, h_t=None, feats_len=None, prox_pos=None, candis_len=None):
        (lang_output, visn_output), pooled_output, candi_score, action_score, candi_score_bt, visn_feats_h_t, house_logit, lr_h = self.bert(
            mode, input_ids, token_type_ids, attention_mask,
            visual_feats, visual_attention_mask, h_t, feats_len, prox_pos, candis_len)

        if mode == 'lang':
            return pooled_output, lang_output
        else:
            return visn_feats_h_t, pooled_output, candi_score, action_score, candi_score_bt, house_logit, lr_h


class LXRTFeatureExtraction(BertPreTrainedModel):
    """
    BERT model for classification.
    """
    def __init__(self, config, mode='lxr'):
        """

        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.bert = LXRTModel(config)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,
                visual_attention_mask=None):
        feat_seq, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                            visual_feats=visual_feats,
                                            visual_attention_mask=visual_attention_mask)
        if 'x' == self.mode:
            return pooled_output
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
            return feat_seq, pooled_output
        elif 'l' in self.mode or 'r' in self.mode:
            return feat_seq

class Progressor(nn.Module):
    def __init__(self, hidden_size):
        super(Progressor, self).__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size+1024+128, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, vfeat, h_ls, cur_p, cur_l):
        bsz = vfeat.size(0)
        new_cur_p = []
        next_turn_logit = []
        for i in range(bsz):
            
            cur_score = self.score(torch.cat([vfeat[i], h_ls[i, cur_p[i], :]], dim=-1))

            if cur_p[i] == cur_l[i]:
                next_turn_logit.append(torch.cat([cur_score, cur_score], dim=-1))
                continue

            
            next_score = self.score(torch.cat([vfeat[i], h_ls[i, cur_p[i]+1, :]], dim=-1))

            next_turn_logit.append(torch.cat([cur_score, next_score], dim=-1))



        return None, torch.stack(next_turn_logit)

class Critic(nn.Module):
    def __init__(self, dropout):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

def cal_norm(feat):
        return torch.sqrt((feat**2).sum(-1)).unsqueeze(-1)



