import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier
from torch.utils.data import DataLoader

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=datasets['eval'], batch_size=args.batch_size, shuffle=False)
    total_data = SeqClsDataset((data['train']+data['eval']), vocab, intent2idx, args.max_len)
    total_loader = DataLoader(dataset=total_data, batch_size=args.batch_size, shuffle=True)


    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    use_gpu = torch.cuda.is_available()
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
                               dropout=args.dropout, bidirectional=args.bidirectional, num_class=len(intent2idx))

    learning_rate = args.lr

    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    if use_gpu:
        model = model.cuda()

    # TODO: init optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        optimizer = adjust_learning_rate(optimizer, epoch)
        epoch_loss_sum = 0
        correct = 0
        loss_history = []
        for i, data in enumerate(total_loader):
            txt = data['text']
            intent = data['intent']
            b1 = torch.tensor(total_data.vocab.encode_batch(txt), dtype=torch.long)
            b2 = torch.tensor(total_data.vocab.encode_batch1(txt), dtype=torch.long)
            x = torch.cat((b1,b2),1)
            y = torch.tensor([total_data.label2idx(t) for t in intent], dtype=torch.long)

            if (use_gpu):
                x,y = x.cuda(),y.cuda()
            model_out = model(x)
            loss = loss_fn(model_out , y)
            epoch_loss_sum += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(model_out, 1)

            correct += (predicted == y).float().sum()

        accuracy = correct / len(total_data)
#         print("Training Accuracy = {}".format(accuracy))

        loss_history.append(epoch_loss_sum)

    # TODO: Inference on test set
    torch.save(model.state_dict(), str(args.ckpt_dir / "best.pt"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
