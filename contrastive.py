import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import BertModel, BertConfig, BertTokenizer, AdamW
from datasets import load_dataset

import nltk
nltk.download('punkt')

# set device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 768
TEMPERATURE = 0.1
BATCH_SIZE = 8
CONTEXT_WINDOW = 3
NEG_SIZE = 10

class BertEncoder(nn.Module):
  def __init__(self):
      super(BertEncoder, self).__init__()
      # Initiate bert model from huggingface
      self.configuration = BertConfig()
      self.bert_model = BertModel(self.configuration)
      self.bert_model.train()

  def forward(self, sent1, attn_mask1):
    # N x (window_size + negative examples) x max_length
    batch_size = sent1.shape[0]
    example_size = sent1.shape[1]
    seq_length = sent1.shape[2]
    sent1 = sent1.reshape(-1, seq_length)
    attn_mask1 = attn_mask1.reshape(-1, seq_length)

    output = self.bert_model(sent1, attention_mask=attn_mask1)

    # Pooling
    embedding1 = torch.mean(
        output['last_hidden_state'][:, 1:, :], axis=1)  # N x hidden_size
    embedding1 = embedding1.reshape(batch_size, example_size, HIDDEN_SIZE)

    #return 1
    return embedding1

def pack_pos_and_neg_sents(sents_list, start_end_list, context_size, sample_size):
  encoded_inputs = np.array(sents_list['input_ids'])
  attn_masks = np.array(sents_list['attention_mask'])

  max_length = encoded_inputs[0].shape[0]
  # A list of packed positive and negative sents, N x (window_size + sample_size) x max_length
  result_inputs = np.empty((0, context_size + sample_size, max_length))
  result_attn_masks = np.empty((0, context_size + sample_size, max_length))

  document_index = 0
  start, end = start_end_list[document_index]
  window_size = context_size // 2

  for i in range(len(encoded_inputs)):
    # Only take nearby sents if the nearby sents are within the same document
    if i >= start+window_size and i < end-window_size:
      # Put anchor in position 0
      nearby_inputs = np.vstack(
          (encoded_inputs[i], encoded_inputs[i-window_size], encoded_inputs[i+window_size]))
      nearby_attn_masks = np.vstack(
          (attn_masks[i], attn_masks[i-window_size], attn_masks[i+window_size]))

      # Negative examples
      # Current negative sampling policy: randomly choose sents from different documents
      negative_inds = np.random.choice(np.concatenate((np.arange(0, start), np.arange(end, len(encoded_inputs)))),
                                       sample_size, replace=False)
      negative_inputs = encoded_inputs[negative_inds]
      pos_neg_inputs = np.append(nearby_inputs, negative_inputs, axis=0)
      negative_attn_masks = attn_masks[negative_inds]
      pos_neg_attn_masks = np.append(
          nearby_attn_masks, negative_attn_masks, axis=0)

      result_inputs = np.append(
          result_inputs, np.expand_dims(pos_neg_inputs, axis=0), axis=0)
      result_attn_masks = np.append(
          result_attn_masks, np.expand_dims(pos_neg_attn_masks, axis=0), axis=0)

    # Next document
    if i == end-window_size and document_index < len(start_end_list)-1:
      document_index = document_index + 1
      start, end = start_end_list[document_index]

  return {
      'sent_input_ids': result_inputs.tolist(),
      'sent_attention_masks': result_attn_masks.tolist()
  }

"""
collate_fn: dynamically set each batch to the same sequence length
"""
def collate_fn(batch):
  sent1 = [(lambda e: torch.stack(e['sent_input_ids']))(e) for e in batch]
  attn_mask1 = [(lambda e: torch.stack(e['sent_attention_masks']))(e)
                for e in batch]

  example_size = sent1[0].shape[0]
  lengths = [(lambda s: s.shape[1])(s) for s in sent1]
  max_length = max(lengths)

  for i in range(len(sent1)):
    seq_length = sent1[i].shape[1]
    if seq_length != max_length:
      # Zero pad to example_size x MAX_LENGTH
      sent1[i] = torch.cat((sent1[i], torch.zeros(
          (example_size, max_length-seq_length))), 1)
      attn_mask1[i] = torch.cat(
          (attn_mask1[i], torch.zeros((example_size, max_length-seq_length))), 1)

  sent1 = torch.stack(sent1, dim=0)
  attn_mask1 = torch.stack(attn_mask1, dim=0)

  # N x example_size x MAX_LENGTH
  return {
      'sent_input_ids': sent1,
      'sent_attention_masks': attn_mask1
  }


def tokenize_sentences(example_batch):
  # Tokenize data using bert tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  article_list = example_batch['article']
  sentenced_list = [(lambda e: nltk.sent_tokenize(e))(e) for e in article_list]

  # Compute star and end index of each document
  start = 0
  start_end_list = []
  for a in sentenced_list:
    end = start + len(a)
    start_end_list.append((start, end))
    start = end

  # Tokenize the flattend sent lists
  merged = list(itertools.chain(*sentenced_list))
  encoded_sents = tokenizer(merged, padding=True, truncation=True)

  # Construct anchor, positive and negative sentence examples
  encoded_batch = pack_pos_and_neg_sents(
      encoded_sents, start_end_list, CONTEXT_WINDOW, NEG_SIZE)  # N x window_size

  return encoded_batch


def train(model, optimizer, train_loader, params):
  train_losses = []

  batch_size = params["batch_size"]
  num_epochs = params["num_epochs"]
  num_iters_per_eval = params["num_iters_per_eval"]
  temperature = params['temperature']

  pos_num = CONTEXT_WINDOW // 2
  neg_sum = NEG_SIZE

  for e in range(num_epochs):
      for i, data in enumerate(train_loader):
          sent1 = data['sent_input_ids'].long().to(device)
          attn_mask1 = data['sent_attention_masks'].long().to(device)
          batch_size = sent1.shape[0]

          # Train batch
          model.zero_grad()
          output = model(sent1, attn_mask1)  # N x example_size x hidden_size

          # Contrastive Loss
          anchor = output[:, 0, :].reshape(batch_size, 1, -1)  # Nx1xH
          scores = torch.bmm(output, torch.transpose(
              anchor, 1, 2))  # N x example_size x1
          scores = torch.squeeze(scores[:, 1:, :], 2)  # remove anchor scores
          scores = scores / temperature
          log_prob = logsoftmax(scores)  # N x sample_size
          loss = -torch.sum(log_prob[:, :pos_num], 1) / pos_num
          loss = torch.mean(loss)

          loss.backward()
          optimizer.step()
          
          # Bookkeep
          print('Train Loss: %.04f' % (loss.item()))
          train_losses.append(loss.item())
          
          # Clear Cache
          torch.cuda.empty_cache()
  
  return train_losses


if __name__ == "__main__":
    # Data
    batch_size = BATCH_SIZE
    full_dataset = load_dataset("cnn_dailymail", '3.0.0')
    dataset = full_dataset['validation'] # Just validation set for dev
    smaller_dataset = dataset.filter(lambda e, i: i < 100, with_indices=True)

    #encoded_dataset.save_to_disk("path/of/my/dataset/directory")

    # Encode and prepare data
    encoded_dataset = smaller_dataset.map(
        tokenize_sentences, batched=True, remove_columns=smaller_dataset.column_names, batch_size=batch_size)

    # Convert to PyTorch Dataloader
    encoded_dataset.set_format(type='torch',
                              columns=['sent_input_ids', 'sent_attention_masks'])
    encoded_dataloader = torch.utils.data.DataLoader(
        encoded_dataset, batch_size=batch_size, collate_fn=collate_fn)


    hidden_size = HIDDEN_SIZE
    num_epochs = 1
    num_iters_per_eval = 10
    temperature = 0.1

    params = {
        "batch_size": batch_size,
        "num_iters_per_eval": num_iters_per_eval,
        "num_epochs": num_epochs,
        "temperature": TEMPERATURE,
    }

    # Model
    model = BertEncoder().to(device)
    print(model)
    optimizer = AdamW(model.parameters(), lr=0.00001)

    # Logsoftmax function
    logsoftmax = nn.LogSoftmax(dim=1)

    # Train
    train_losses = train(
        model, optimizer, encoded_dataloader, params)

    # Plot
    plt.figure()
    plt.title('Train Loss per ' + str(num_iters_per_eval) + ' iterations')
    plt.plot(train_losses)
    plt.xlabel('ith ' + str(num_iters_per_eval) + ' iterations')
    plt.ylabel('Train Loss')
