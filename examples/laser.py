import logging
import os
import sys

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import LASER
PATH_TO_LASER = '../../LASER'
BYTE_PAIR_ENCODINGS = PATH_TO_LASER + '/models/93langs.fcodes'
LASER_MODEL = PATH_TO_LASER + '/models/bilstm.93langs.2018-12-26.pt'

os.environ['LASER'] = PATH_TO_LASER
sys.path.insert(0, PATH_TO_LASER)

sys.path.insert(0, PATH_TO_LASER + '/source')
sys.path.insert(0, PATH_TO_LASER + '/source/lib')
from embed import SentenceEncoder
from text_processing import TokenLine, BPEfastLoad, BPEfastApplyLine

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

os.environ['CUDA_VISIBLE_DEVICES'] = ''


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    bpe_batch = []
    for sent in batch:
        tokens = TokenLine(sent, lang='en', lower_case=True, romanize=False)
        bpe_codes = BPEfastApplyLine(tokens, params_senteval['bpe'])
        bpe_batch += [bpe_codes]
    embeddings = params['laser'].encode_sentences(bpe_batch)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 128}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['bpe'] = BPEfastLoad('', BYTE_PAIR_ENCODINGS)
params_senteval['laser'] = SentenceEncoder(model_path=LASER_MODEL, max_sentences=None, max_tokens=None, cpu=True,
                                           fp16=False, verbose=False, sort_kind='quicksort')

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
    results = se.eval([transfer_tasks])
    print(results)
