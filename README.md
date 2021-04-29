# COS548-Final_Project
## hyperparams: lambda, temperature, pos+neg examples
1. baseline: lambda: 0.0, temperature:N/A
2. SCL#1: lambda:0.5, temperature: 1.0, pos:3, neg:3
3. SCL#2: lambda:0.5, temperature: 1.0, pos:3, neg: all neg in batch
4. AUG#1: lambda:0.5, temperature N/A, pos:3, neg:3

| Tables            | Glove         | Baseline      | SCL#1     | AUG#1    | SCL#2    | 
| ------------------|:-------------:| -------------:| ---------:| --------:| --------:|
| SNLI eval         |               |  78.6         | 75.07     | 80.71    | 73.23  |
| STS12             | 44.09         |  66.06        | 64.10     | 65.85    | 65.05  |
| STS13             | 43.02         |  63.41        | 63.81     | 62.76    | 68.34  |
| STS14             | 47.43         |  67.03        | 66.13     | 66.20    | 69.08  |
| STS15             | 50.08         |  74.46        | 74.11     | 73.10    | 76.09  |
| STS16             | 40.30         |  67.11        | 67.69     | 64.97    | 70.22  |
| (AVG) STS         | 44.984        |  67.614       | 67.168    | 66.576   | 69.76  |
| MR                | 51.14         |  66.1         | 64.76     |          |        | 
| CR                | 78.78         |  80.34        | 80.45     |
| SUBJ              | 99.6          |  99.62        | 99.6      |
| MPQA              | 87.59         |  86.92        | 86.78     |
| SST Binary        | 79.68         |  79.08        | 75.45     |
| SST Fine-grained  | 43.8          |  38.96        | 42.31     |
| TREC              | 82.8          |  83.8         | 89.2      |
| MRPC              | 70.78         |  69.86        | 72.12     |
| (AVG) Transfer    |    