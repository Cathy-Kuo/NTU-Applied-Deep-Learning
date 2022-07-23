import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from pathlib import Path
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from dataset import MultiChoiceDataset
from torch.nn.utils.rnn import pad_sequence
import random
from transformers import AdamW
import os

from transformers import BertTokenizerFast, BertTokenizer, BertForMultipleChoice
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForMultipleChoice

TRAIN = "train"
DEV = "public"
SPLITS = [TRAIN, DEV]

def cal_acc(label_pred, label_true):
    c = 0
    for i in range(len(label_pred)):
        if (label_pred[i] == label_true[i]):
            c += 1
    return c


def collate(samples) :
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    masks_tensors = [s[2] for s in samples]
    label_tensor = torch.stack([s[3] for s in samples])

    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    masks_tensors = pad_sequence(masks_tensors, batch_first=True)

    return tokens_tensors, segments_tensors, masks_tensors, label_tensor

def main(args):
    model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-macbert-base")
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
    
    train_path = args.data_dir
    data = {}
    data['train'] = json.loads(train_path.read_text())
    
    path = args.context_dir
    context = json.loads(path.read_text())

    train_data = []
    for item in data['train']:
        qs = []
        ps = []
        paragraphs = item['paragraphs']
        for i in range(5):
            qs.append(item['question'])
        if len(paragraphs) > 5:
            relevent = item['relevant']
            label = paragraphs.index(relevent)
            for i in range(4):
                ps.append(context[paragraphs[i]])
            if label > 4:
                label = 4
                ps.append(context[relevent])
            else:
                ps.append(context[paragraphs[4]])
        else:
            pad_num = 5-len(paragraphs)
            relevent = item['relevant']
            label = paragraphs.index(relevent)
            for p in paragraphs:
                ps.append(context[p])
            paragraphs.remove(relevent)
            for i in range(pad_num):
                ps.append(context[paragraphs[i%len(paragraphs)]])
        train_data.append([qs, ps, label])




    trainset = MultiChoiceDataset(data=train_data, tokenizer=tokenizer)

    BATCH_SIZE = 1
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=1e-5)
    EPOCHS = 5

    accumulation_steps = 32
    
    epoch_pbar = trange(EPOCHS, desc="Epoch")
    for epoch in epoch_pbar:

        running_loss = 0.0
        acc = 0
        model.train()
        model.zero_grad()                                   # Reset gradients tensors
        for i, data in tqdm(enumerate(trainloader)):
            if use_gpu:
                tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t.cuda() for t in data]
    #            tokens_tensors, masks_tensors, label_tensors = [t.cuda() for t in data]
            else:
                tokens_tensors, segments_tensors, masks_tensors, label_tensors = [t for t in data]
    #            tokens_tensors, masks_tensors, label_tensors = [t for t in data]

            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=label_tensors)

            loss = outputs[0]                               # Compute loss function
            loss = loss / accumulation_steps                # Normalize our loss (if averaged)
            loss.backward()                                 # Backward pass
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()                           # Reset gradients tensors     

            label_preds = torch.argmax(outputs[1], dim=1)
            acc += cal_acc(label_preds, label_tensors)

            # 紀錄當前 batch loss
            running_loss += loss.item()

        print('epoch: ', epoch + 1,  ', loss: ',running_loss, ', acc: ', acc/len(train_data))
        

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
        default="./data/train.json",
    )
    parser.add_argument(
        "--context_dir",
        type=Path,
        help="Directory to the context.",
        default="./data/context.json",
    )
    parser.add_argument(
        "--model_out_dir",
        type=Path,
        help="Directory to the model.",
        default="./ckpt/context",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.model_out_dir.mkdir(parents=True, exist_ok=True)
    main(args)
