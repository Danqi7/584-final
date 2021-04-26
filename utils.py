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

from models import SentBert

# set device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 512
NUM_CLASS = 3
BATCH_SIZE = 64

ENTAILMEN_LABEL = 0
NEUTRAL_LABEL = 1
CONTRADICTION_LABEL = 2

def tokenize_sentences(example_batch):
    # Tokenize data using bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded_sent1 = tokenizer(
        example_batch['premise'], padding=True, truncation=True)
    sent1_input_ids = encoded_sent1['input_ids']
    sent1_token_type_ids = encoded_sent1['token_type_ids']
    sent1_attn_mask = encoded_sent1['attention_mask']

    encoded_sent2 = tokenizer(
        example_batch['hypothesis'], padding=True, truncation=True)
    sent2_input_ids = encoded_sent2['input_ids']
    sent2_token_type_ids = encoded_sent2['token_type_ids']
    sent2_attn_mask = encoded_sent2['attention_mask']

    return {'sent1_input_ids': sent1_input_ids,
            'sent1_token_type_ids': sent1_token_type_ids,
            'sent1_attention_mask': sent1_attn_mask,
            'sent2_input_ids': sent2_input_ids,
            'sent2_token_type_ids': sent2_token_type_ids,
            'sent2_attention_mask': sent2_attn_mask}


def load_snli_data(type, batch_size, save_dir):
  dataset = load_dataset("snli", split=type)

  # Filter out data examples with -1 label
  dataset = dataset.filter(lambda e: e['label'] >= 0)

  # Shuffle
  # Don't shuffle since we want same premise cluster together
  #dataset = dataset.shuffle()

  # Tokenize data using bert tokenizer, default batch size is 1000
  encoded_dataset = dataset.map(
      tokenize_sentences, batched=True, batch_size=batch_size)

  # Convert to PyTorch Dataloader
  encoded_dataset.set_format(type='torch',
                             columns=['sent1_input_ids', 'sent1_attention_mask',
                                      'sent2_input_ids', 'sent2_attention_mask',
                                      'label'])

  # Save pre-processed data to disk
  encoded_dataset.save_to_disk(save_dir)

  encoded_dataloader = torch.utils.data.DataLoader(
      encoded_dataset, batch_size=batch_size)

  return encoded_dataloader
