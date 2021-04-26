import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, AutoTokenizer, AutoModel
from datasets import load_dataset
from datasets import load_from_disk

# set device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 512
NUM_CLASS = 3

ENTAILMEN_LABEL = 0
NEUTRAL_LABEL = 1
CONTRADICTION_LABEL = 2

class SentBert(nn.Module):
    def __init__(self, input_dim, output_dim, tokenizer):
        super(SentBert, self).__init__()
        # Initiate bert model from huggingface
        self.bert_model = AutoModel.from_pretrained(
            "google/bert_uncased_L-8_H-512_A-8")
        self.bert_model.train()

        # Linear Layers
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.linear2 = nn.Linear(input_dim//2, output_dim)

        # Tokenizer
        self.tokenizer = tokenizer

    def forward(self, sent1, attn_mask1, sent2, attn_mask2):
        '''
            sent1: (N x T1)
            sent2: (N x T2)
            attn_mask1: (N x T1)
            attn_mask2: (N x T2)
        '''
        # N x T x hidden_size
        N, T1 = sent1.shape
        _, T2 = sent2.shape
        out1 = self.bert_model(sent1, attention_mask=attn_mask1)
        out2 = self.bert_model(sent2, attention_mask=attn_mask2)
        H = out1['last_hidden_state'].shape[-1]

        # Pooling
        # TODO: Ablation study: consider [CLS]/[SEP] in pooling?
        hidden_states1 = out1['last_hidden_state']  # (N x T1 x H)
        hidden_states1 = hidden_states1 * torch.reshape(attn_mask1, (N, T1, 1))
        hidden_states2 = out2['last_hidden_state']
        hidden_states2 = hidden_states2 * torch.reshape(attn_mask2, (N, T2, 1))
        embedding1 = torch.mean(hidden_states1[:, 1:, :], axis=1)  # N x H
        embedding2 = torch.mean(hidden_states2[:, 1:, :], axis=1)

        # embedding1 = torch.mean(out1['last_hidden_state'][:,1:,:], axis=1) # N x hidden_size
        # embedding2 = torch.mean(out2['last_hidden_state'][:,1:,:], axis=1)

        # Concate embeddings (u, v, |u-v|)
        # TODO: Ablation study different concate methods
        diff = torch.abs(embedding1-embedding2)
        merged = torch.cat((embedding1, embedding2, diff), -1)

        merged = self.linear1(merged)
        merged = self.linear2(merged)  # N x class

        return merged, (embedding1, embedding2)

    def encode(self, sents):
        # TODO: check if .eval() is needed
        #  Answer: yes
        self.bert_model.eval()
        self.eval()

        with torch.no_grad():
            encoded_sent1 = self.tokenizer(
                sents, padding=True, truncation=True)
            input_ids = torch.Tensor(encoded_sent1['input_ids']).long()
            attn_mask = torch.Tensor(encoded_sent1['attention_mask']).long()
            out = self.bert_model(input_ids, attention_mask=attn_mask)
            embeddings = torch.mean(
                out['last_hidden_state'][:, 1:, :], axis=1)  # N x hidden_size

        return embeddings.detach()
