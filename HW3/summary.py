import torch
torch.cuda.is_available()
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import collections

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tw_rouge import get_rouge


from tqdm import trange

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from dataset import NLGDataset
from torch.nn.utils.rnn import pad_sequence
import random
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from transformers import AdamW, Adafactor
import os


def collate(samples) :
    text_ids = [s[0] for s in samples]
    text_masks = [s[1] for s in samples]
    label_ids = [s[2] for s in samples]
    label_masks = [s[3] for s in samples]

    text_ids = pad_sequence(text_ids, batch_first=True)
    text_masks = pad_sequence(text_masks, batch_first=True)
    label_ids = pad_sequence(label_ids, batch_first=True)
    label_masks = pad_sequence(label_masks, batch_first=True)

    return text_ids, text_masks, label_ids, label_masks

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.5 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def main(args):
    train_path = args.data_dir
    data={'train':[]}
    with open(train_path, 'r') as reader:
        for line in reader:
            single_data = json.loads(line)
            del single_data['date_publish']
            del single_data['source_domain']
            del single_data['split']
            data['train'].append(single_data)
    
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    
    trainset = NLGDataset(data=data['train'], tokenizer=tokenizer)
    BATCH_SIZE = 1
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    EPOCHS = 30
    learning_rate = 3e-4
    accumulation_steps = 32
    optimizer = Adafactor(model.parameters(), lr=learning_rate, relative_step=False)
    
    for epoch in tqdm(range(EPOCHS)):

        optimizer = adjust_learning_rate(optimizer, epoch, learning_rate)
        running_loss = 0.0
        model.train()
        model.zero_grad() 
        optimizer.zero_grad() 
        for i, data in tqdm(enumerate(trainloader)):
            if use_gpu:
                text_ids, text_masks, label_ids, label_masks = [t.cuda() for t in data]
            else:
                text_ids, text_masks, label_ids, label_masks = [t for t in data]

            outputs = model(input_ids=text_ids, 
                            attention_mask=text_masks, 
                            labels=label_ids)

            loss = outputs.loss
            loss = loss / accumulation_steps  
            loss.backward()              
            if (i+1) % accumulation_steps == 0:  
                optimizer.step() 
                optimizer.zero_grad()

            running_loss += loss.item()

        print('epoch: ', epoch + 1,  ', loss: ',running_loss)
    

    output_dir = args.model_out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_to_save = model.module if hasattr(model, 'module') else model 
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/train.jsonl",
    )
    parser.add_argument(
        "--model_out_dir",
        type=Path,
        help="Directory to the model.",
        default="./ckpt/best",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.model_out_dir.mkdir(parents=True, exist_ok=True)
    main(args)
