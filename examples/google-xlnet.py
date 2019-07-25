import logging
import os
import sys
import json

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import XLNET
PATH_TO_XLNET = 'xlnet'
os.environ['xlnet'] = PATH_TO_XLNET
sys.path.insert(0, PATH_TO_XLNET)
import abstract_xlnet

# xlnet params
MODEL_CONFIG_PATH = '/xlnet_config.json'
SPIECE_MODEL_FILE = '/spiece.model'
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
    embeddings = abstract_xlnet.encode_sentences(batch, params['xlnet_config']['model_config'],
                                                 params['xlnet_config']['run_config'],
                                                 params['xlnet_config']['tokenizer'],
                                                 params['xlnet_max_seq_length'])
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 8}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['xlnet_config'] = abstract_xlnet.build_xlnet_config(MODEL_CONFIG_PATH, SPIECE_MODEL_FILE)
params_senteval['xlnet_max_seq_length'] = MAX_SEQ_LENGTH

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
    results = se.eval(transfer_tasks)
    print(json.dumps(results, indent=4, sort_keys=True))
