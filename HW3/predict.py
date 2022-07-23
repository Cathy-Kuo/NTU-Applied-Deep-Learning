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

import torch
from tqdm import trange

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from dataset import NLGDataset1
from torch.nn.utils.rnn import pad_sequence
import random
from transformers import T5Tokenizer, MT5ForConditionalGeneration

def collate(samples) :
    text_ids = [s[0] for s in samples]
    text_masks = [s[1] for s in samples]

    text_ids = pad_sequence(text_ids, batch_first=True)
    text_masks = pad_sequence(text_masks, batch_first=True)

    return text_ids, text_masks

def main(args):
    model_dir = Path("./ckpt/best")
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    
    data_path = args.input_file
    data={'public':[]}
    with open(data_path, 'r') as reader:
        for line in reader:
            single_data = json.loads(line)
            d = {'maintext': single_data['maintext'], 'id': single_data['id']}
            data['public'].append(d)
    
    

    publicset = NLGDataset1(data=data['public'], tokenizer=tokenizer)

    BATCH_SIZE = 1
    publicloader = DataLoader(publicset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        
    label_pred = []
    label_true = []
    answer = []
    model.eval()
    with torch.no_grad():
        for i, datas in tqdm(enumerate(publicloader)):
            if use_gpu:
                text_ids, text_masks = [t.cuda() for t in datas]
            else:
                text_ids, text_masks = [t for t in datas]


            generated_ids = model.generate(
                input_ids = text_ids,
                attention_mask = text_masks, 
                max_length=120, 
                num_beams=3,
                top_k=30, 
                do_sample=True,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            ans = {"title":preds[0], "id": data['public'][i]['id']}
            answer.append(ans)

    out = str(args.output_file)
    with open(args.output_file, mode="w") as writer: 
        for line in answer:
            j = json.dumps(line)
            writer.write(j+'\n')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Directory to the preprocessed caches.",
        required=True
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
