# ADL NTU 109 Spring Homework 2

## How to train my model -- Context
python3.8 train_context.py --data_dir (training data) --context_dir (context data) --model_out_dir (model output direction)

## How to train my model -- Question Answering
python3.8 train_QA.py --data_dir (training data) --context_dir (context data) --model_out_dir (model output direction)

## for example
python3.8 train_QA.py --data_dir "./data/train.json" --context_dir "./data/context.json" --model_out_dir "QA"
