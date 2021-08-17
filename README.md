# Sentence Embeddings using Supervised Contrastive Learning
This code accompanies the paper: [Sentence Embeddings using Supervised Contrastive Learning](https://arxiv.org/abs/2106.04791)

Danqi Liao.

This is a final project for [COS 584](https://princeton-nlp.github.io/cos484/cos584.html).

![scl](https://github.com/Danqi7/584-final/blob/master/illustrates.png)

### To download data(SNLI) from HuggingFace and train:
```
python bert_sent_embed.py --pos_num -1 --neg_num -1 --use_SCL
```


### To load data from local disk and train:
```
python bert_sent_embed.py --load_data_from_disk --pos_num -1 --neg_num -1 --use_SCL
```

### To evaluate trained model on downstream sentence tasks through [SentEval](https://github.com/facebookresearch/SentEval)
1. ```cd SentEval-master/examples/```
2. ```Modify 'sentbert_eval.py' to change $MODEL_PATH to your model```
3.  ```Modify 'sentbert_eval.py' to change $transfer_tasks to the tasks you want to evaluate```
4. Run the evaluation script 
```
python sentbert_eval.py
```

### To generate sentence embedding using trained model
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = SentBert(512*3, 3, tokenizer)
model.load_state_dict(torch.load(model_path, map_location=device))

embedding = model.encode("hello world.")
```

### Evaluation results on Avg Glove embeddings, our SBERT baseline, all positive/negative SCL model with lambda 0.3

| Tables                      | STS(12-16) AVG         | Sentence Transfer Tasks AVG       |
| ----------------------------|:-------------------:   | :--------------------------------:| 
| Avg. GloVe Embeddings       | 44.98                  |  74.27                            |
| Our SBERT baseline          | 67.61                  |  75.56                            |
| allpalln-lambda0.3-SCL      | 70.44                  |  76.16                            |

Note, our SBERT baseline is not the full scale SBERT model from [SBERT](https://arxiv.org/abs/1908.10084), but rather our
own replementation using only SNLI data, medium-sized [8/512 bert](https://github.com/google-research/bert), and the same hyperparameters 
with the SCL models. The reason for using a smaller sized bert and only SNLI data is simply computation constraints.

### Example trained sentence encoding model
[sent-bert8](https://drive.google.com/drive/folders/1TD0R0Y7uoV3cUv4SqeMrBTHjmTqmK8NY?usp=sharing)
