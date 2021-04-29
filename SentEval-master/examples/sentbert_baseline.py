from __future__ import absolute_import, division, unicode_literals
import torch
import sys
import io
import numpy as np
import logging
import time

from transformers import BertModel, BertConfig, BertTokenizer, AdamW

# import SentEval
PATH_TO_SENTEVAL = '../'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# import bert_sent_embed
sys.path.insert(0, '../../')
from models import SentBert

#from sentence_transformers import SentenceTransformer

# Set PATHs
PATH_TO_DATA = '../data'
MODEL_PATH = '../models/warmup/sent-bert8.pt'

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 512
OUTPUT_DIM = 3

def init_model(model_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentBert(HIDDEN_SIZE*3, OUTPUT_DIM, tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params['sentbert'].encode(batch)

    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

encoder = init_model(MODEL_PATH)
params_senteval['sentbert'] = encoder

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    start_time = time.time()
    results = se.eval(transfer_tasks)
    #print(results)
    elapsed_time = time.time() - start_time
    print('Evaluation Finished in : ', time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time)))
