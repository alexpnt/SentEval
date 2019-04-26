import logging
import os
import sys

import numpy as np

import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

os.environ['CUDA_VISIBLE_DEVICES'] = ''


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]

    embeddings = []
    for sent in batch:
        indexed_tokens = params['tokenizer'].encode(sent)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            hidden_states, past = params['openai-gpt-2'](tokens_tensor)
            hidden_states = hidden_states.numpy()
            emb = np.amax(hidden_states[0], axis=0)
            embeddings += [emb]
    return np.vstack(embeddings)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 1024}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['tokenizer'] = GPT2Tokenizer.from_pretrained('gpt2')
params_senteval['openai-gpt-2'] = GPT2Model.from_pretrained('gpt2')
params_senteval['openai-gpt-2'].eval()

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'STSBenchmarkUnsupervised',
                      'SICKRelatednessUnsupervised','Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(['STSBenchmarkUnsupervised', 'SICKRelatednessUnsupervised'])
    print(results)
