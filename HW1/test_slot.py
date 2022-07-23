import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import TagClsDataset
from model import TagClassifier
from model import TagClassifier1
from utils import Vocab
from torch.utils.data import DataLoader

def pred_to_label(predicted,mask,dataset):
    batch_size = len(predicted)
    batch_label = []
    for b in range(batch_size):
        k = int(torch.sum(mask[b]))
        labels = [dataset.idx2label(p) for p in predicted[b]]
        labels = labels[:k]
        labels = ' '.join(labels)
        batch_label.append(labels)
    
    return batch_label

def loss_fn(outputs, labels):
    labels = labels.view(-1)  
    mask = (labels >= 0).float()
    num_tokens = int(torch.sum(mask))
    outputs = outputs[range(outputs.shape[0]), labels]*mask
    return -torch.sum(outputs)/num_tokens, mask

def cal_accuracy(out, y, mask):
    batch_size = len(y)
    correct = 0
    for b in range(batch_size):
        same = 0
        k = int(torch.sum(mask[b]))
        for j in range(k):
            if out[b][j] == y[b][j]:
                same += 1
        if same == k:
            correct += 1
    
    return correct

def prepare_label(tags, length, dataset):
    y = []
    for i in range(len(tags)):
        tag = tags[i].split()
        y1 = [dataset.label2idx(t) for t in tag]
        pad = length-len(y1)
        for p in range(pad):
            y1.append(-1)
        y.append(y1)
        
    y = torch.tensor(y, dtype=torch.long)
    return y

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = TagClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = TagClassifier1(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    pd_ids = [data[i]['id'] for i in range(len(data))]
    pd_slots = []
    for i, data in enumerate(test_loader):
        tokens = data['token']
        tags = data['tag']
        word_token = torch.tensor(dataset.vocab.encode_batch1(tokens), dtype=torch.long)
        char_token = torch.tensor(dataset.vocab.encode_char(tokens), dtype=torch.long)
        y = prepare_label(tags, len(word_token[0]), dataset)
        if (use_gpu):
            word_token,char_token,y = word_token.cuda(),char_token.cuda(),y.cuda()
            
        model_out = model(word_token, char_token)
        loss, mask = loss_fn(model_out , y)
        
        batch_size = word_token.size(0)
        _, predicted = torch.max(model_out, 1)
        predicted = predicted.view(batch_size, -1).cpu().numpy()
        mask = mask.view(batch_size, -1).cpu()

        ans = pred_to_label(predicted, mask, dataset)
        pd_slots += ans
        
    import csv

    path = args.pred_file
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'tags'])
        for i in range(len(pd_ids)):
            writer.writerow([pd_ids[i], pd_slots[i]])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
