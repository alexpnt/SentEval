# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division

import os
import sys
import logging
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
import json

tf.logging.set_verbosity(0)

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# tensorflow session
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TFHUB_CACHE_DIR'] = '../data/models/tfhub_modules'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]

    # Set up graph.
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        use_multilingual_16_embed = hub.Module(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1")
        embedded_text = use_multilingual_16_embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Initialize session.
    session = tf.Session(graph=g)
    session.run(init_op)

    # Compute embeddings.
    embeddings = session.run(embedded_text, feed_dict={text_input: batch})

    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(['STSBenchmarkUnsupervised', 'SICKRelatednessUnsupervised'])
    print(json.dumps(results, indent=4, sort_keys=True))
