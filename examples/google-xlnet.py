import logging
import os
import sys
import json

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import XLNET
PATH_TO_XLNET = '/home/arpinto/WIT/projects/proofs-of-concept/xlnet'
os.environ['xlnet'] = PATH_TO_XLNET
sys.path.insert(0, PATH_TO_XLNET)
import xlnet_embed

# xlnet params
MODEL_BASE_PATH = '/home/arpinto/WIT/data/models/xlnet/xlnet_cased_L-12_H-768_A-12/'
MODEL_CONFIG_PATH = MODEL_BASE_PATH + 'xlnet_config.json'
MODEL_CKPT_PATH = MODEL_BASE_PATH + 'xlnet_model.ckpt'
MODEL_FINETUNED_DIR = MODEL_BASE_PATH + 'finetuned/'
SPIECE_MODEL_FILE = MODEL_BASE_PATH + 'spiece.model'

MAX_SEQ_LENGTH = 512

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = xlnet_embed.encode_sentences(batch, params['xlnet_tokenizer'],
                                              params['xlnet_config']['max_seq_length'],
                                              params['xlnet_config']['model_config_path'],
                                              params['xlnet_config']['model_ckpt_path'],
                                              params['xlnet_config']['model_finetuned_dir'])
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 8}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['xlnet_tokenizer'] = xlnet_embed.tokenize_fn_builder(SPIECE_MODEL_FILE)
params_senteval['xlnet_config'] = {'max_seq_length': MAX_SEQ_LENGTH,
                                   'model_base_path': MODEL_BASE_PATH,
                                   'model_config_path': MODEL_CONFIG_PATH,
                                   'model_ckpt_path': MODEL_CKPT_PATH,
                                   'model_finetuned_dir': MODEL_FINETUNED_DIR}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'STSBenchmarkUnsupervised',
                      'SICKRelatednessUnsupervised', 'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(['STS13'])
    print(json.dumps(results, indent=4, sort_keys=True))
