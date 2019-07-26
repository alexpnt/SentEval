import logging
import os
import sys
import json
import numpy as np
import torch
from pytorch_transformers import XLNetModel, XLNetTokenizer

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = []
    for sent in batch:
        input_ids = torch.tensor(params['xlnet_tokenizer'].encode(sent)).unsqueeze(0)
        with torch.no_grad():
            outputs = params_senteval['xlnet_model'](input_ids)
            last_hidden_states_batch = outputs[0].numpy()  # first entry of the output tuple
            last_hidden_states = last_hidden_states_batch[0]
            # emb = np.amax(last_hidden_states, axis=0)
            emb = last_hidden_states[-1]
            embeddings += [emb]
    return np.vstack(embeddings)


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 64}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['xlnet_tokenizer'] = XLNetTokenizer.from_pretrained('xlnet-base-cased')
params_senteval['xlnet_model'] = XLNetModel.from_pretrained('xlnet-base-cased')
params_senteval['xlnet_model'] = params_senteval['xlnet_model'].eval()


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
    results = se.eval(['STSBenchmarkUnsupervised'])
    print(json.dumps(results, indent=4, sort_keys=True))
