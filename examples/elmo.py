from __future__ import absolute_import, division

import logging
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

tf.logging.set_verbosity(0)

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TFHUB_CACHE_DIR'] = '../data/models/tfhub_modules'


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]

    elmo = params['elmo'](batch, signature="default", as_dict=True)["elmo"].eval(session=params['tf_session'])

    # elmo_mean_pooling = params['elmo'](batch, signature="default", as_dict=True)["default"].eval(
    #     session=params['tf_session'])
    # elmo_max_pooling = np.amax(elmo, axis=1)

    return elmo


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 128}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['elmo'] = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
params_senteval['tf_session'] = tf.Session()
params_senteval['tf_session'].run(tf.global_variables_initializer())

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'Length', 'WordContent', 'Depth',
                      'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
