import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
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
    pd_ids = []
    pd_intents = []
    for i, data in enumerate(test_loader):
        txt = data['text']
        ids = data['id']
        b1 = torch.tensor(dataset.vocab.encode_batch(txt), dtype=torch.long)
        b2 = torch.tensor(dataset.vocab.encode_batch1(txt), dtype=torch.long)
        x = torch.cat((b1,b2),1)
        if (use_gpu):
            x = x.cuda()

        model_out = model(x)
        _, predicted = torch.max(model_out, 1)
        pd_ids += ids
        predicted = predicted.cpu().detach().numpy()
        intents = [dataset.idx2label(p) for p in predicted]
        pd_intents += intents

    # TODO: write prediction to file (args.pred_file)
    import csv

    path = args.pred_file
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'intent'])
        for i in range(len(pd_ids)):
            writer.writerow([pd_ids[i], pd_intents[i]])


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
        default="./cache/intent/",
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
