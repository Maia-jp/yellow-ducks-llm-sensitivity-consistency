# Model comparison

Models: DeepSeek-R1-7B, Mixtral-8x7B

## All results

| Model | Dataset | Strategy | Samples | Sensitivity | Consistency | Micro-F1 | N/A share |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-7B | CB | simple | 250 | 0.277 | 0.679 | 0.740 | 0.002 |
| Mixtral-8x7B | CB | simple | 250 | 0.253 | 0.561 | 0.600 | 0.000 |
| DeepSeek-R1-7B | CB | instruct | 250 | 0.299 | 0.678 | 0.733 | 0.003 |
| Mixtral-8x7B | CB | instruct | 250 | 0.179 | 0.622 | 0.703 | 0.000 |
| DeepSeek-R1-7B | CB | fewshot | 250 | 0.297 | 0.644 | 0.704 | 0.003 |
| Mixtral-8x7B | CB | fewshot | 250 | 0.185 | 0.559 | 0.626 | 0.000 |
| DeepSeek-R1-7B | TREC | simple | 500 | 0.167 | 0.735 | 0.783 | 0.002 |
| Mixtral-8x7B | TREC | simple | 500 | 0.191 | 0.657 | 0.729 | 0.004 |
| DeepSeek-R1-7B | TREC | instruct | 500 | 0.164 | 0.739 | 0.774 | 0.005 |
| Mixtral-8x7B | TREC | instruct | 500 | 0.192 | 0.683 | 0.706 | 0.008 |
| DeepSeek-R1-7B | TREC | fewshot | 500 | 0.135 | 0.763 | 0.804 | 0.001 |
| Mixtral-8x7B | TREC | fewshot | 500 | 0.143 | 0.646 | 0.738 | 0.004 |
| Mixtral-8x7B | RTE | simple | 2490 | 0.260 | 0.595 | 0.693 | 0.218 |
| Mixtral-8x7B | RTE | instruct | 2490 | 0.099 | 0.751 | 0.850 | 0.009 |
| Mixtral-8x7B | RTE | fewshot | 2490 | 0.060 | 0.755 | 0.824 | 0.004 |
| Mixtral-8x7B | DBPedia | simple | 2000 | 0.049 | 0.816 | 0.865 | 0.000 |
| Mixtral-8x7B | DBPedia | instruct | 2000 | 0.054 | 0.811 | 0.868 | 0.002 |
| Mixtral-8x7B | DBPedia | fewshot | 2000 | 0.073 | 0.768 | 0.843 | 0.005 |

## Datasets present for all models: CB, TREC

| Dataset | Strategy | Consistency DeepSeek-R1-7B | Consistency Mixtral-8x7B | Micro-F1 DeepSeek-R1-7B | Micro-F1 Mixtral-8x7B | N/A share DeepSeek-R1-7B | N/A share Mixtral-8x7B | Sensitivity DeepSeek-R1-7B | Sensitivity Mixtral-8x7B |
|---|---|---|---|---|---|---|---|---|---|
| CB | simple | 0.679 | 0.561 | 0.740 | 0.600 | 0.002 | 0.000 | 0.277 | 0.253 |
| CB | instruct | 0.678 | 0.622 | 0.733 | 0.703 | 0.003 | 0.000 | 0.299 | 0.179 |
| CB | fewshot | 0.644 | 0.559 | 0.704 | 0.626 | 0.003 | 0.000 | 0.297 | 0.185 |
| TREC | simple | 0.735 | 0.657 | 0.783 | 0.729 | 0.002 | 0.004 | 0.167 | 0.191 |
| TREC | instruct | 0.739 | 0.683 | 0.774 | 0.706 | 0.005 | 0.008 | 0.164 | 0.192 |
| TREC | fewshot | 0.763 | 0.646 | 0.804 | 0.738 | 0.001 | 0.004 | 0.135 | 0.143 |

