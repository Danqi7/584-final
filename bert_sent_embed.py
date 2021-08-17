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
from utils import load_snli_data, eval

# set device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 512
NUM_CLASS = 3

ENTAILMEN_LABEL = 0
NEUTRAL_LABEL = 1
CONTRADICTION_LABEL = 2


def train(model, optimizer, scheduler, loss_function, train_loader, eval_data, params):
    # Params
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    num_iters_per_eval = params['num_iters_per_eval']
    temperature = params['temperature']
    use_SCL = params['use_SCL']
    temperature = params['temperature']
    lamb = params['lamb']
    positive_num = params['pos_num']
    negative_num = params['neg_num']

    # Print some info about train data
    num_data = len(train_loader) * batch_size
    num_iterations_per_epoch = len(train_loader)
    print('Total number of Iterations: ', num_iterations_per_epoch * num_epochs)
    print("num_data: ", num_data, "\nnum_iters_per_epoch: ",
          num_iterations_per_epoch)

    # Format sample eval data
    eval_sent1 = eval_data['sent1_input_ids']
    eval_sent2 = eval_data['sent2_input_ids']
    eval_attn_mask1 = eval_data['sent1_attention_mask']
    eval_attn_mask2 = eval_data['sent2_attention_mask']
    eval_labels = eval_data['label']

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
            # Positive examples only exist if there is entrailment pair
            if use_SCL == True:
                SCL_cnt = 0
                batch_size = sent1.shape[0]  # N

                SCLLoss = 0
                for eidx in range(batch_size):
                    # positive examples only exist if there is entrailment pair
                    if labels[eidx] == ENTAILMEN_LABEL:
                        SCL_cnt += 1
                        current_premise = sent1[eidx, :]
                        anchor = torch.unsqueeze(embeds1[eidx], dim=1)  # H x 1
                        # 1 x H, its entailment is pos
                        scores = torch.unsqueeze(embeds2[eidx], dim=0)

                        pos_cnt = 1
                        neg_cnt = 0
                        same_premise_idxs = []
                        entailment_idxs = [eidx]
                        pos_candidates_idxs = []  # w/t itself

                        # figure out entailment_idxs, pos_candidates_idxs
                        for j in range(batch_size):
                            if torch.all(sent1[j, :] == current_premise):
                                same_premise_idxs.append(j)
                                if j != eidx and labels[j] == ENTAILMEN_LABEL:
                                    entailment_idxs.append(j)
                                    pos_candidates_idxs.append(j)

                        # positive examples                        
                        if positive_num > 0:
                            current_positive_num = positive_num - 1
                            if len(pos_candidates_idxs) < positive_num - 1 : # not enough positive, take whatever we have
                                current_positive_num = len(pos_candidates_idxs)
                            pos_idxs = np.random.choice(pos_candidates_idxs, current_positive_num, replace=False)
                        else:
                            pos_idxs = pos_candidates_idxs
                        for pos_id in pos_idxs:
                            scores = torch.cat(
                                (scores, torch.unsqueeze(embeds2[pos_id, :], dim=0)), dim=0)
                            pos_cnt += 1

                        # negative examples
                        candidates_idxs = np.arange(batch_size)
                        candidates_idxs = np.delete(candidates_idxs, entailment_idxs)
                        all_neg_num = len(candidates_idxs)
                        if negative_num > 0: 
                            neg_idxs = np.random.choice(candidates_idxs, all_neg_num, replace=False)
                        else: # take all negative examples
                            neg_idxs = candidates_idxs
                        for neg_id in neg_idxs:
                            # ((pos_cnt + neg_cnt) x H)
                            scores = torch.cat(
                                (scores, torch.unsqueeze(embeds2[neg_id, :], dim=0)), dim=0)
                            neg_cnt += 1
                        logits = scores @ anchor  # (pos+neg) x 1
                        logits = logits / temperature
                        log_prob = F.log_softmax(logits, dim=0)  # (pos+neg) x 1
                        SCLLoss += -torch.sum(log_prob[:pos_cnt, :]) / pos_cnt

            loss = loss_function(output, labels)
            
            if use_SCL == True:
                loss = (1-lamb) * loss + lamb * (SCLLoss / SCL_cnt)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Evaluate on sample validation dataset
            if i % num_iters_per_eval == 0 or e == num_epochs - 1 and i == num_iterations_per_epoch - 1:
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

                    sample_out, _ = model(sample_sent1, sample_attn1,
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
    parser.add_argument("--store_files", type=str, default="./models/",
                        help="Where to store the trained model")
    parser.add_argument("--batch_size", default=64,
                        type=int, help="How many sentence pairs in a batch")
    parser.add_argument("--num_epochs", default=1,
                        type=int, help="epochs to train")
    parser.add_argument("--load_data_from_disk", default=False, action='store_true',
                        help="Whether to load train/val data from disk or load from HF repo")
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for softmax")
    parser.add_argument("--use_SCL", default=False, action='store_true', help="Whether to use SCL Loss in addition to CE Loss")
    parser.add_argument("--lamb", default=0.5, type=float, help="lambda for SCL Loss")
    parser.add_argument("--pos_num", default=3, type=int, help="Positive Example number for super contrastive learning")
    parser.add_argument("--neg_num", default=3, type=int, help="Negative example number for super contrastive learning")

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

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=1000)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10000)
        
        print('Done loading datasets. train data%s\n validation data%s\n test data%s\n' % (
            train_dataset, validation_dataset, test_dataset))
    else:
        print("Loading data from scratch (HG) ...")
        train_dataloader = load_snli_data('train', batch_size, save_dir="./train")
        test_dataloader = load_snli_data('test', 10000, save_dir='./test')
        validation_dataloader = load_snli_data('validation', 10000, save_dir='./validation')

    sample_eval_data = next(iter(validation_dataloader)) # Just for eval model during training
    print(sample_eval_data['label'].shape)

    # Hyperparams
    hidden_size = HIDDEN_SIZE
    num_class = NUM_CLASS
    num_iters_per_print = 10
    num_epoch_per_eval = 1
    learning_rate = 2e-5
    warmup_step = int((len(train_dataloader) * 0.1))
    print('warmup_step: ', warmup_step)
    params = {
        "batch_size": batch_size,
        "num_iters_per_print": num_iters_per_print,
        "num_epochs": args.num_epochs,
        "num_iters_per_eval": 10,
        "save_file": args.store_files,
        "load_data_from_disk": args.load_data_from_disk,
        "use_SCL": args.use_SCL,
        "temperature": args.temperature,
        "lamb": args.lamb,
        "pos_num": args.pos_num,
        "neg_num": args.neg_num,
    }

    # Model
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
    model = SentBert(hidden_size * 3, num_class, tokenizer).to(device)
    print(model)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = transformers.get_constant_schedule_with_warmup(
        optimizer, warmup_step)


    # Cross Entropy Loss
    loss_function = nn.CrossEntropyLoss()

    start_time = time.time()
    # Train
    train_losses, validation_losses, train_accs, validation_accs = train(
        model, optimizer, scheduler, loss_function, train_dataloader, sample_eval_data, params)
    
    # Create model directory
    dir_name = args.store_files + "scl%d-lr%s-lamb%.1f-t%.1f-%.1f" % (
        args.use_SCL, learning_rate, args.lamb, args.temperature, start_time)
    os.mkdir(dir_name)
    print('Saving model to dir: ', dir_name)

    model_save_name = 'sent-bert8.pt'
    path = dir_name + "/" + model_save_name

    # Save Model
    torch.save(model.state_dict(), path)

    # Eval on full testing data
    print('Evaluating on %d test data...'%(test_dataloader))

    final_acc = eval(model, test_dataloader)
    print("Full Testing Accuracy: ", final_acc)

    f = open(dir_name + "/model_info.txt", "a")
    content = "model: " + dir_name + "\n" + "Train Loss: " + \
        str(np.mean(np.array(train_losses))) + "\nValidation loss: " + \
        str(np.mean(np.array(validation_losses)))
    content += "\nTrain Accuracy: " + \
        str(np.mean(train_accs)) + "\nTest Accuracy: " + str(final_acc)
    content += "\nlr: " + str(learning_rate) + "\nbatch size: " + \
        str(batch_size) + "\nnum_epochs: " + str(args.num_epochs)
    content += "\nArguments: %s" %(args)
    content += "\nArchitecture: " + model.__str__()
    f.write(content)
    f.close()
