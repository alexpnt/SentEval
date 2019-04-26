import logging
import os
import sys

import fastBPE
import numpy as np
import torch

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

PATH_TO_XLM = '../../XLM'
XLM_MODEL = '../../../../data/models/xlm/mlm_tlm_xnli15_1024.pth'

PATH_TO_LASER = '../../LASER'

BPE_CODES = '../../../../data/models/xlm/codes_xnli_15'
BPE_VOCAB = '../../../../data/models/xlm/vocab_xnli_15'


# import XLM
sys.path.insert(0, PATH_TO_XLM)
from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

# import LASER (needed for tokenization tools)
os.environ['LASER'] = PATH_TO_LASER
sys.path.insert(0, PATH_TO_LASER + '/source/lib')
from text_processing import TokenLine, BPEfastApplyLine

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

    # tokenize and encode sentences in BPE format
    for sent in batch:
        tokens = TokenLine(sent, lang='en', lower_case=True, romanize=False)
        bpe_sent = BPEfastApplyLine(tokens, params_senteval['bpe'])
        bpe_sent = '</s> {} </s>'.format(bpe_sent.strip()).split()
        bpe_batch += [bpe_sent]

    batch_size = params['batch_size']
    max_length = max([len(bpe_sent) for bpe_sent in bpe_batch])

    # create input data
    word_ids = torch.LongTensor(max_length, batch_size).fill_(params['xlm_params'].pad_index)
    for i in range(len(bpe_batch)):
        sent = torch.LongTensor([dico.index(w) for w in bpe_batch[i][0]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in bpe_batch])
    langs = torch.LongTensor([params['xlm_params'].lang2id[lang] for lang in ['en']]).unsqueeze(0).expand(max_length, batch_size)

    # encode
    embeddings = []
    with torch.no_grad():
        tensor = params['xlm']('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
        emb = np.amax(tensor.numpy(), axis=0)
        embeddings += [emb]
    print(np.vstack(embeddings).shape)
    return np.vstack(embeddings)


def load_xlm():
    model = torch.load(XLM_MODEL)
    params = AttrDict(model['params'])
    dico = Dictionary(model['dico_id2word'], model['dico_word2id'], model['dico_counts'])

    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    return model, dico, params

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 128}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

model, dico, params = load_xlm()
params_senteval['xlm'] = TransformerModel(params, dico, True, True)
params_senteval['xlm'].load_state_dict(model['model'])

params_senteval['xlm_dico'] = dico
params_senteval['xlm_params'] = params

params_senteval['bpe'] = fastBPE.fastBPE(BPE_CODES, BPE_VOCAB)

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
    print(results)
