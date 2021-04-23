from __future__ import absolute_import, division, unicode_literals
import sys
import io
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# import SentEval
PATH_TO_SENTEVAL = '../'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set PATHs
PATH_TO_DATA = '../data'

def prepare(params, samples):
    return


def batcher(params, batch):
    #print('====before====: ', batch)
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    #print(batch)
    embeddings = params['sentbert'].encode(batch)

    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
params_senteval['sentbert'] = encoder

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['STS12']
    results = se.eval(transfer_tasks)
    #print(results)
