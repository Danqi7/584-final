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
  encoded_dataset = dataset.map(tokenize_sentences, batched=True, batch_size=batch_size)

  # Convert to PyTorch Dataloader
  encoded_dataset.set_format(type='torch',
                             columns=['sent1_input_ids', 'sent1_attention_mask',
                                      'sent2_input_ids', 'sent2_attention_mask', 
                                      'label'])
  
  # Save pre-processed data to disk
  encoded_dataset.save_to_disk(save_dir)

  encoded_dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size)

  return encoded_dataloader

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
        # TODO: need to apply attn_mask to zero out padded
        # TODO: Ablation study: consider [CLS]/[SEP] in pooling?
        hidden_states1 = out1['last_hidden_state'] # (N x T1 x H)
        hidden_states1 = hidden_states1 * torch.reshape(attn_mask1, (N,T1,1))
        hidden_states2 = out2['last_hidden_state']
        hidden_states2 = hidden_states2 * torch.reshape(attn_mask2, (N,T2,1))
        embedding1 = torch.mean(hidden_states1[:, 1:, :], axis=1) # N x H
        embedding2 = torch.mean(hidden_states2[:, 1:, :], axis=1) 

        # embedding1 = torch.mean(out1['last_hidden_state'][:,1:,:], axis=1) # N x hidden_size
        # embedding2 = torch.mean(out2['last_hidden_state'][:,1:,:], axis=1)

        # Concate embeddings (u, v, |u-v|)
        # TODO: Ablation study different concate methods
        diff = torch.abs(embedding1-embedding2)
        merged = torch.cat((embedding1, embedding2, diff), -1)

        merged = self.linear1(merged)
        merged = self.linear2(merged) # N x class

        return merged, (embedding1, embedding2)

    def encode(self, sents):
        # TODO: check if .eval() is needed
        #  Answer: yes
        self.bert_model.eval()
        self.eval()

        with torch.no_grad():
            encoded_sent1 = self.tokenizer(sents, padding=True, truncation=True)
            input_ids = torch.Tensor(encoded_sent1['input_ids']).long()
            attn_mask = torch.Tensor(encoded_sent1['attention_mask']).long()
            out = self.bert_model(input_ids, attention_mask=attn_mask)
            embeddings = torch.mean(
                out['last_hidden_state'][:, 1:, :], axis=1)  # N x hidden_size
    
        return embeddings.detach()


def train(model, optimizer, loss_function, train_loader, eval_data, params):
    # Params
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    num_iters_per_print = params['num_iters_per_print']
    num_epoch_per_eval = params['num_epoch_per_eval']
    save_file = params['save_file']
    load_data_from_disk = params['load_data_from_disk']
    temperature = params['temperature']
    use_SCL = params['use_SCL']

    # Print some info about train data
    num_data = len(train_loader) * batch_size
    num_iters_per_epoch = num_data / batch_size
    print('Total number of Iterations: ', num_iters_per_epoch * num_epochs)
    print("num_data: ", num_data, "\nnum_iters_per_epoch: ", num_iters_per_epoch)

    # Format sample eval data
    print(len(eval_data['sent1_input_ids']))
    print(eval_data['sent1_input_ids'].shape)
    # if load_data_from_disk != True:
    #     eval_sent1 = torch.stack(eval_data['sent1_input_ids'], dim=0).permute(1, 0)  # n x T
    #     eval_sent2 = torch.stack(eval_data['sent2_input_ids'],dim=0).permute(1, 0) # n x T
    #     eval_attn_mask1 = torch.stack(eval_data['sent1_attention_mask'], dim=0).permute(1,0)
    #     eval_attn_mask2 = torch.stack(eval_data['sent2_attention_mask'], dim=0).permute(1,0)
    # else:
    print(eval_data.keys())
    # eval_premise = eval_data['premise'][:3, :]
    
    eval_sent1 = eval_data['sent1_input_ids']
    eval_sent2 = eval_data['sent2_input_ids']
    eval_attn_mask1 = eval_data['sent1_attention_mask']
    eval_attn_mask2 = eval_data['sent2_attention_mask']
    eval_labels = eval_data['label']
    #print(eval_sent1[:6, :])
    #print("?=? , ", torch.all(eval_sent1[0, :] == eval_sent1[1, :]))

    eval_num = eval_labels.shape[0]
    print(
        'eval_labels.shape: ', eval_labels.shape
    )

    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    for e in range(num_epochs):
        for i, data in enumerate(train_loader):
            #if load_data_from_disk != True:
            #     sent1 = torch.stack(data['sent1_input_ids'], dim=0)  # T x n
            #     sent1 = sent1.permute(1, 0).to(device) # n x T
            #     attn_mask1 = torch.stack(data['sent1_attention_mask'],dim=0)
            #     attn_mask1 = attn_mask1.permute(1, 0).to(device)

            #     sent2 = torch.stack(data['sent2_input_ids'],dim=0)
            #     sent2 = sent2.permute(1, 0).to(device) # n x T
            #     attn_mask2 = torch.stack(data['sent2_attention_mask'],dim=0)
            #     attn_mask2 = attn_mask2.permute(1, 0).to(device)
            # else:
            sent1 = data['sent1_input_ids'].to(device)
            sent2 = data['sent2_input_ids'].to(device)
            attn_mask1 = data['sent1_attention_mask'].to(device)
            attn_mask2 = data['sent2_attention_mask'].to(device)
            labels = data['label'].to(device)

            # Train batch
            model.zero_grad()
            # output: (N x classes), embeddings: {embed1, embed2} both (N x hidden_size)
            output, (embeds1, embeds2) = model(sent1, attn_mask1, sent2, attn_mask2)

            # Supervised Contrastive Loss
            # TODO: numerical stability? exp(x) / sum(exp(x))
            if use_SCL == True:
                negative_num = 3  # TODO: hyperparam
                SCLLoss = 0
                for i in range(sent1.shape[1]):
                    label = labels[i]
                    if label == ENTAILMEN_LABEL:
                        current_premise = sent1[i, :]
                        #scores = embeds1[i]
                        anchor = embeds1[i]
                        scores = embeds2[i] # 1 x H
                        pos_cnt = 1
                        neg_cnt = 0
                        for j in range(sent1.shape[1]):
                            # positive example
                            if torch.all(sent1[j, :] == current_premise) and labels[j] == ENTAILMEN_LABEL:
                                scores = torch.cat((scores, embeds2[j]),dim=0) # {pos_num x H}
                                pos_cnt += 1
                            elif neg_cnt < negative_num:
                                # negative examples
                                scores = torch.cat((scores, embeds2[j]),dim=0) # ((pos_cnt + neg_cnt) x H)
                                neg_cnt += 1
                        print('computing SCL...')
                        anchor = torch.unsqueeze(anchor, dim=1) # H x 1
                        logits = scores @ anchor # (pos+neg) x 1
                        logits = logits / temperature
                        log_prob = F.log_softmax(logits,dim=0)  # (pos+neg) x 1
                        SCLLoss += -torch.sum(log_prob[:pos_cnt, :]) / pos_cnt


            # CE Loss                    
            loss = loss_function(output, labels)
            
            if use_SCL == True:
                loss += SCLLoss
            
            loss.backward()
            optimizer.step()

            # Evaluate on sample validation dataset
            if (e % num_epoch_per_eval == 0 and i == num_iters_per_epoch - 1) or (e == num_epochs-1 and i == num_iters_per_epoch - 1):
                with torch.no_grad():
                    sample_size = 100
                    if eval_num > sample_size:
                        sample_mask = np.random.choice(
                            eval_num, sample_size, replace=False)
                        sample_mask = torch.from_numpy(sample_mask)
                        sample_sent1 = torch.index_select(eval_sent1, 0, sample_mask)
                        sample_sent2 = torch.index_select(eval_sent2, 0, sample_mask)
                        sample_attn1 = torch.index_select(
                            eval_attn_mask1, 0, sample_mask)
                        sample_attn2 = torch.index_select(
                            eval_attn_mask2, 0, sample_mask)
                        sample_label = torch.index_select(
                            eval_labels, 0, sample_mask)
                    else:
                        sample_sent1 = eval_sent1
                        sample_sent2 = eval_sent2
                        sample_attn1 = eval_attn_mask1
                        sample_attn2 = eval_attn_mask2
                        sample_label = eval_labels
                    
                    # Move to device
                    sample_sent1 = sample_sent1.to(device)
                    sample_sent2 = sample_sent2.to(device)
                    sample_attn1 = sample_attn1.to(device)
                    sample_attn2 = sample_attn2.to(device)
                    sample_label = sample_label.to(device)

                    sample_out = model(sample_sent1, sample_attn1,
                                        sample_sent2, sample_attn2) # N x 3
                    sample_loss = loss_function(sample_out, sample_label)
                    sample_pred = torch.argmax(sample_out, 1)
                    sample_acc = (sample_pred == sample_label).sum().item() / sample_label.shape[0]
                    
                    output_pred = torch.argmax(output, 1)
                    acc = (output_pred == labels).sum().item() / labels.shape[0]
                    train_losses.append(loss)
                    eval_losses.append(sample_loss)
                    train_accs.append(acc)
                    eval_accs.append(sample_acc)

                    print('[%d/%d][%d/%d]\tTrain Loss: %.4f\tEval Loss: %.4f\tTrain Acc: %.4f\tEval Acc: %.4f'
                        % (e, num_epochs, i, len(train_loader),
                            loss.item(), sample_loss.item(), acc, sample_acc))
            
            # Clear Cache
            torch.cuda.empty_cache()
    
    return train_losses, eval_losses, train_accs, eval_accs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_files", type=str, required=True,
                        help="Where to store the trained model")
    parser.add_argument("--batch_size", default=64,
                        type=int, help="How many sentence pairs in a batch")
    parser.add_argument("--num_epochs", default=1,
                        type=int, help="epochs to train")
    parser.add_argument("--load_data_from_disk", default=False, action='store_true',
                        help="Whether to load train/val data from disk or load from HF repo")
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for softmax")
    parser.add_argument("--use_SCL", default=False, action='store_true', help="Whether to use SCL Loss in addition to CE Loss")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.store_files):
        os.makedirs(args.store_files)
    
    # Data
    batch_size = args.batch_size
    if args.load_data_from_disk == True:
        print("Loading data from disk...")
        train_dataset = load_from_disk("./train/")
        validation_dataset = load_from_disk("./validation/")
        test_dataset = load_from_disk("./test/")
        #premise = test_dataset['premise']
        #print('%s'%(type(premise)))

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10000)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000)

        print('Done loading datasets. train data%s\n validation data%s\n test data%s\n' % (
            train_dataset, validation_dataset, test_dataset))
    else:
        print("Loading data from scratch (HG) ...")
        train_dataloader = load_snli_data('train', batch_size, save_dir="./train")
        test_dataloader = load_snli_data('test', 10000, save_dir='./test')
        validation_dataloader = load_snli_data('validation', 10000, save_dir='./validation')

    sample_eval_data = next(iter(validation_dataloader)) # Just for eval model during training

    hidden_size = HIDDEN_SIZE
    num_class = NUM_CLASS
    num_epochs = args.num_epochs
    num_iters_per_print = 10
    num_iters_per_eval = 10
    num_epoch_per_eval = 1
    temperature = args.temperature
    params = {
        "batch_size": batch_size,
        "num_iters_per_eval": num_iters_per_eval,
        "num_iters_per_print": num_iters_per_print,
        "num_epochs": num_epochs,
        "num_epoch_per_eval": num_epoch_per_eval,
        "save_file": args.store_files,
        "load_data_from_disk": args.load_data_from_disk,
        "temperature": 1,
        "use_SCL": args.use_SCL,
    }
    # Model
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
    model = SentBert(hidden_size * 3, num_class, tokenizer).to(device)
    print(model)
    # TODO: add warmup steps + weight decaying similar to the paper
    optimizer = AdamW(model.parameters(), lr=0.00001)

    # Cross Entropy Loss
    loss_function = nn.CrossEntropyLoss()

    # Train
    train_losses, validation_losses, train_accs, validation_accs = train(
        model, optimizer, loss_function, train_dataloader, sample_eval_data, params)

    # Eval on full testing data
    eval_data = next(iter(test_dataloader))
    print('Evaluating on %d test data...'%(test_dataloader))
    with torch.no_grad():
        # if args.load_data_from_disk == True:
        #     eval_sent1 = torch.stack(
        #         eval_data['sent1_input_ids'], dim=0).permute(1, 0).to(device)  # n x T
        #     eval_sent2 = torch.stack(
        #         eval_data['sent2_input_ids'], dim=0).permute(1, 0).to(device)  # n x T
        #     eval_attn_mask1 = torch.stack(
        #         eval_data['sent1_attention_mask'], dim=0).permute(1, 0).to(device)
        #     eval_attn_mask2 = torch.stack(
        #         eval_data['sent2_attention_mask'], dim=0).permute(1, 0).to(device)
        # else:
        eval_sent1 = eval_data['sent1_input_ids'].to(device)
        eval_sent2 = eval_data['sent2_input_ids'].to(device)
        eval_attn_mask1 = eval_data['sent1_attention_mask'].to(device)
        eval_attn_mask2 = eval_data['sent2_attention_mask'].to(device)
        eval_labels = eval_data['label'].to(device)

        eval_out = model(eval_sent1, eval_attn_mask1, eval_sent2, eval_attn_mask2)  # N x 3
        eval_loss = loss_function(eval_out, eval_labels)
        eval_pred = torch.argmax(eval_out, 1)
        eval_acc = (eval_pred == eval_labels).sum().item() / eval_labels.shape[0]
        print("Full Testing Data Loss: ", eval_losses, "\tTest Accuracy: ", eval_acc)


    # Plot
    plt.figure()
    plt.title('Train Loss per ' + str(num_iters_per_eval) + ' iterations')
    plt.plot(train_losses)
    plt.xlabel('ith ' + str(num_iters_per_eval) + ' iterations')
    plt.ylabel('Train Loss')

    plt.figure()
    plt.title('Validation Loss per ' + str(num_iters_per_eval) + ' iterations')
    plt.plot(validation_losses)
    plt.xlabel('ith ' + str(num_iters_per_eval) + ' iterations')
    plt.ylabel('Validation Loss')

    plt.figure()
    plt.title('Train Accuracy per ' + str(num_iters_per_eval) + ' iterations')
    plt.plot(train_accs)
    plt.xlabel('ith ' + str(num_iters_per_eval) + ' iterations')
    plt.ylabel('Train Accuracy')

    plt.figure()
    plt.title('Validation Accuracy per ' + str(num_iters_per_eval) + ' iterations')
    plt.plot(validation_accs)
    plt.xlabel('ith ' + str(num_iters_per_eval) + ' iterations')
    plt.ylabel('Validation Accuracy')
