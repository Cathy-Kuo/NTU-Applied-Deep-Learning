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
from torch.utils.data import Dataset

from transformers import BertTokenizerFast, BertTokenizer, BertForMultipleChoice
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoTokenizer
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,BertTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

from tqdm.auto import tqdm
import numpy as np
def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size = 20, max_answer_length = 50 ):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k['id']: i for i, k in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        min_null_score = None 
        valid_answers = []
        
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]

    return predictions

def prepare_validation_features(examples, tokenizer, pad_on_right):
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

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def collate_val(samples) :
    tokens_tensors = torch.tensor([s['input_ids'] for s in samples])
    segments_tensors = torch.tensor([s['token_type_ids'] for s in samples])
    masks_tensors = torch.tensor([s['attention_mask'] for s in samples]) 
    
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    masks_tensors = pad_sequence(masks_tensors, batch_first=True)

    return {'input_ids':tokens_tensors, 'token_type_ids':segments_tensors, 'attention_mask':masks_tensors}



def main(args):
    output_dir = Path("./ckpt/context")
    model = BertForMultipleChoice.from_pretrained(output_dir)
    tokenizer = BertTokenizerFast.from_pretrained(output_dir)
    
    data_path = args.test_file
    data_ = {}
    data_['public'] = json.loads(data_path.read_text())
    path = args.context_file
    context = json.loads(path.read_text())
    

    pub_data = []
    pub_data1 = []
    paras = []
    paras1 = []
    for item in data_['public']:
        paragraphs = item['paragraphs']
        question = item['question']
        qid = item['id']
        if len(paragraphs)<6:
            qs = []
            ps = []
            for i in range(5):
                qs.append(item['question'])
            for p in paragraphs:
                ps.append(context[p])
            paras.append(paragraphs)
            pub_data.append([qs, ps, qid])
        else:
            qs1 = []
            qs2 = []
            ps1 = []
            ps2 = []
            for i in range(3):
                qs1.append(item['question'])
                ps1.append(context[paragraphs[i]])
            for i in range(len(paragraphs)-3):
                qs2.append(item['question'])
                ps2.append(context[paragraphs[i+3]])
            paras1.append([paragraphs[:3], paragraphs[3:]])
            pub_data1.append([[qs1, ps1], [qs2, ps2], qid])
            
    label_preds = {}
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        for i, data in tqdm(enumerate(pub_data)):
            qs = data[0]
            ps = data[1]
            qid = data[2]
            tokens = tokenizer(text=qs, text_pair=ps, add_special_tokens=True,max_length=512,truncation=True, return_tensors='pt', padding=True)
            if use_gpu:
                tokens_tensors = pad_sequence([torch.tensor(tokens['input_ids'])], batch_first=True).cuda()
                segments_tensors = pad_sequence([torch.tensor(tokens['token_type_ids'])], batch_first=True).cuda()
                masks_tensors = pad_sequence([torch.tensor(tokens['attention_mask'])], batch_first=True).cuda()
            else:
                tokens_tensors = pad_sequence([torch.tensor(tokens['input_ids'])], batch_first=True)
                segments_tensors = pad_sequence([torch.tensor(tokens['token_type_ids'])], batch_first=True)
                masks_tensors = pad_sequence([torch.tensor(tokens['attention_mask'])], batch_first=True)
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
            pred = torch.argmax(outputs[0], dim=1)
            pid = paras[i][pred]
            label_preds[qid] = pid
            
    with torch.no_grad():
        for i, data in tqdm(enumerate(pub_data1)):
            qs1 = data[0][0]
            qs2 = data[1][0]
            ps1 = data[0][1]
            ps2 = data[1][1]
            qid = data[2]
            tokens1 = tokenizer(text=qs1, text_pair=ps1, add_special_tokens=True,max_length=512,truncation=True, return_tensors='pt', padding=True)
            tokens2 = tokenizer(text=qs2, text_pair=ps2, add_special_tokens=True,max_length=512,truncation=True, return_tensors='pt', padding=True)

            if use_gpu:
                tokens_tensors1 = pad_sequence([torch.tensor(tokens1['input_ids'])], batch_first=True).cuda()
                segments_tensors1 = pad_sequence([torch.tensor(tokens1['token_type_ids'])], batch_first=True).cuda()
                masks_tensors1 = pad_sequence([torch.tensor(tokens1['attention_mask'])], batch_first=True).cuda()
                tokens_tensors2 = pad_sequence([torch.tensor(tokens2['input_ids'])], batch_first=True).cuda()
                segments_tensors2 = pad_sequence([torch.tensor(tokens2['token_type_ids'])], batch_first=True).cuda()
                masks_tensors2 = pad_sequence([torch.tensor(tokens2['attention_mask'])], batch_first=True).cuda()
            else:
                tokens_tensors1 = pad_sequence([torch.tensor(tokens1['input_ids'])], batch_first=True)
                segments_tensors1 = pad_sequence([torch.tensor(tokens1['token_type_ids'])], batch_first=True)
                masks_tensors1 = pad_sequence([torch.tensor(tokens1['attention_mask'])], batch_first=True)
                tokens_tensors2 = pad_sequence([torch.tensor(tokens2['input_ids'])], batch_first=True)
                segments_tensors2 = pad_sequence([torch.tensor(tokens2['token_type_ids'])], batch_first=True)
                masks_tensors2 = pad_sequence([torch.tensor(tokens2['attention_mask'])], batch_first=True)

            outputs1 = model(input_ids=tokens_tensors1, token_type_ids=segments_tensors1, attention_mask=masks_tensors1)
            outputs2 = model(input_ids=tokens_tensors2, token_type_ids=segments_tensors2, attention_mask=masks_tensors2)
            pred1 = torch.argmax(outputs1[0], dim=1)
            m1 = torch.max(outputs1[0], dim=1)
            pid1 = paras1[i][0][pred1]
            pred2 = torch.argmax(outputs2[0], dim=1)
            pid2 = paras1[i][1][pred2]
            m2 = torch.max(outputs2[0], dim=1)

            pid = pid1 if m1>m2 else pid2
            label_preds[qid] = pid

    output_dir = Path("./ckpt/QA")
    model = AutoModelForQuestionAnswering.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    max_length = 512 
    doc_stride = 256
    pad_on_right = tokenizer.padding_side == "right"
    
    val_data = []            
    for item in data_['public']:
        single_data = {'id':item['id'], 'question':item['question'], 'context':context[label_preds[item['id']]]}
        val_data.append(single_data)
    
    expand_val = []

    for item in val_data:
        features = prepare_validation_features(item, tokenizer, pad_on_right)
        for i in range(len(features['input_ids'])):
            single_data = {'input_ids':features['input_ids'][i], 
                           'token_type_ids':features['token_type_ids'][i], 
                           'attention_mask':features['attention_mask'][i],
                           'offset_mapping':features['offset_mapping'][i],
                           'example_id':features['example_id'][0]}
            expand_val.append(single_data)
            
    valset1 = QADataset1(data=expand_val)
    
    
    trainer = Trainer(
        model,
        data_collator=collate_val,
        tokenizer=tokenizer,
    )
    
    raw_predictions_val = trainer.predict(valset1)
    final_predictions1 = postprocess_qa_predictions(val_data, valset1, raw_predictions_val.predictions, tokenizer)
    
    out = str(args.output_file)
    with open(out, "w") as outfile: 
        json.dump(final_predictions1, outfile)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--context_file",
        type=Path,
        help="Directory to the preprocessed caches.",
        required=True
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Directory to the preprocessed caches.",
        required=True
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    s = args.output_file
    s1 = s.split('/')
    s = Path(s[:len(s)-len(s1[-1])-1])
    s.mkdir(parents=True, exist_ok=True)
    main(args)
