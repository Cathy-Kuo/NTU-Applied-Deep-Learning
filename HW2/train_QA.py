import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import collections
import spacy

import torch
from tqdm import trange

from pathlib import Path
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from dataset import MultiChoiceDataset
from dataset import QADataset1
from torch.nn.utils.rnn import pad_sequence
import random

from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,BertTokenizer
from transformers import TrainingArguments, Trainer

TRAIN = "train"
DEV = "public"
SPLITS = [TRAIN, DEV]

def prepare_train_features(examples, tokenizer, pad_on_right):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=512,
        stride=256,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"]
        # If no answers are given, set the cls_index as answer.
        start_char = answers[0]["start"]
        end_char = start_char + len(answers[0]["text"])

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def collate(samples) :
    tokens_tensors = torch.tensor([s['input_ids'] for s in samples])
    segments_tensors = torch.tensor([s['token_type_ids'] for s in samples])
    masks_tensors = torch.tensor([s['attention_mask'] for s in samples]) 
    start_tensor = torch.tensor([s['start_positions'] for s in samples])
    end_tensor = torch.tensor([s['end_positions'] for s in samples])
    
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    masks_tensors = pad_sequence(masks_tensors, batch_first=True)

    return {'input_ids':tokens_tensors, 'token_type_ids':segments_tensors, 'attention_mask':masks_tensors, 'start_positions':start_tensor, 'end_positions':end_tensor}

def main(args):
    model = AutoModelForQuestionAnswering.from_pretrained('hfl/chinese-macbert-base')
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
    
    train_path = args.data_dir
    data = {}
    data['train'] = json.loads(train_path.read_text())
    
    path = args.context_dir
    context = json.loads(path.read_text())

    train_data = []
    for item in data['train']:
        ans = item['answers']
        single_data = {'id':item['id'], 'question':item['question'], 'answers':ans, 'context':context[item['relevant']]}
        train_data.append(single_data)
        
    max_length = 512 
    doc_stride = 256 
    pad_on_right = tokenizer.padding_side == "right"

    expand_train = []
    for item in train_data:
        features = prepare_train_features(item, tokenizer, pad_on_right)
        for i in range(len(features['input_ids'])):
            single_data = {'input_ids':features['input_ids'][i], 
                           'token_type_ids':features['token_type_ids'][i], 
                           'attention_mask':features['attention_mask'][i],
                           'start_positions':features['start_positions'][i],
                           'end_positions':features['end_positions'][i]}
            expand_train.append(single_data)
    expand_val = expand_train[:1000]
    expand_train = expand_train[1000:]
    trainset = QADataset1(data=expand_train)
    valset = QADataset1(data=expand_val)
    
    args1 = TrainingArguments(
        f"QA_ckpt",
        evaluation_strategy = "epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        save_strategy="epoch",
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model,
        args1,
        train_dataset=trainset,
        eval_dataset = valset,
        data_collator=collate,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    out_path = args.model_out_dir
    
    trainer.save_model(out_path)



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
