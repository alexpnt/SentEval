import logging
import os
import sys

import numpy as np
import tensorflow as tf
from bert import tokenization, extract_features, modeling

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

BERT_MODEL = '../../../../data/models/bert/base/en/uncased_L-12_H-768_A-12'
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# SentEval prepare and batcher
def prepare(params, samples):
    return


def pooling(hidden_states, layer_index=0, strategy='CLS'):
    if strategy == 'min':
        embedding = np.amin(hidden_states[f'layer_output_{layer_index}'], axis=0).flat
    elif strategy == 'max':
        embedding = np.amax(hidden_states[f'layer_output_{layer_index}'], axis=0).flat
    elif strategy == 'mean':
        embedding = hidden_states[f'layer_output_{layer_index}'].mean(axis=0).flat
    elif strategy == 'min-max':
        m1 = np.amin(hidden_states[f'layer_output_{layer_index}'], axis=0)
        m2 = np.amax(hidden_states[f'layer_output_{layer_index}'], axis=0)
        embedding = np.concatenate((m1, m2)).flat
    elif strategy == 'mean-max':
        m1 = hidden_states[f'layer_output_{layer_index}'].mean(axis=0)
        m2 = np.amax(hidden_states[f'layer_output_{layer_index}'], axis=0)
        embedding = np.concatenate((m1, m2)).flat
    else:
        # CLS token vector
        embedding = hidden_states['layer_output_0'][0].flat
    return embedding


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]

    unique_id = 0
    examples = []
    for sent in batch:
        unicode_sent = tokenization.convert_to_unicode(sent)
        examples += [extract_features.InputExample(unique_id=unique_id, text_a=unicode_sent, text_b=None)]
        unique_id += 1

    features = extract_features.convert_examples_to_features(examples=examples, seq_length=MAX_SEQ_LENGTH,
                                                             tokenizer=params['tokenizer'])

    input_fn = extract_features.input_fn_builder(features=features, seq_length=MAX_SEQ_LENGTH)
    embeddings = []
    for result in params['bert'].predict(input_fn, yield_single_examples=True):
        embeddings += [pooling(result, layer_index=0, strategy='max')]

    return np.vstack(embeddings)


def load_bert():
    bert_config = modeling.BertConfig.from_json_file(BERT_MODEL + '/bert_config.json')
    layer_indexes = [-1, -2]

    run_config = tf.contrib.tpu.RunConfig(tpu_config=tf.contrib.tpu.TPUConfig())

    model_fn = extract_features.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=BERT_MODEL + '/bert_model.ckpt',
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=BATCH_SIZE)

    return estimator


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 128}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['bert'] = load_bert()
params_senteval['tokenizer'] = tokenization.FullTokenizer(vocab_file=BERT_MODEL + '/vocab.txt', do_lower_case=True)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'STSBenchmarkUnsupervised',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(['STSBenchmarkUnsupervised'])
    print(results)
