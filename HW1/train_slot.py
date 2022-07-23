import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import TagClsDataset
from utils import Vocab
from model import TagClassifier
from model import TagClassifier1
from model import TagClassifier2

from torch.utils.data import DataLoader

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


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

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, TagClsDataset] = {
        split: TagClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=datasets['eval'], batch_size=args.batch_size, shuffle=False)
    total_data = TagClsDataset((data['train']+data['eval']), vocab, tag2idx, args.max_len)
    total_loader = DataLoader(dataset=total_data, batch_size=args.batch_size, shuffle=True)


    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    use_gpu = torch.cuda.is_available()
    model = TagClassifier1(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
                               dropout=args.dropout, bidirectional=args.bidirectional, num_class=len(tag2idx))

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

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        epoch_loss_sum = 0
        correct = 0
        loss_history = []
        for i, data in enumerate(train_loader):
            tokens = data['token']
            tags = data['tag']
            word_token = torch.tensor(datasets['train'].vocab.encode_batch1(tokens), dtype=torch.long)
            char_token = torch.tensor(datasets['train'].vocab.encode_char(tokens), dtype=torch.long)
            y = prepare_label(tags, len(word_token[0]), datasets['train'])
            if (use_gpu):
                word_token,char_token,y = word_token.cuda(),char_token.cuda(),y.cuda()
            model_out = model(word_token, char_token)
            loss, mask = loss_fn(model_out , y)
            epoch_loss_sum += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = word_token.size(0)
            _, predicted = torch.max(model_out, 1)
            predicted = predicted.view(batch_size, -1)
            mask = mask.view(batch_size, -1)
            correct += cal_accuracy(predicted, y, mask)


        accuracy = correct / len(datasets['train'])
        print("Training Accuracy = {}".format(accuracy))

        loss_history.append(epoch_loss_sum)

        epoch_loss_sum = 0
        correct = 0
        for i, data in enumerate(valid_loader):
            tokens = data['token']
            tags = data['tag']
            word_token = torch.tensor(datasets['eval'].vocab.encode_batch1(tokens), dtype=torch.long)
            char_token = torch.tensor(datasets['eval'].vocab.encode_char(tokens), dtype=torch.long)
            y = prepare_label(tags, len(word_token[0]), datasets['eval'])
            if (use_gpu):
                word_token,char_token,y = word_token.cuda(),char_token.cuda(),y.cuda()

            model_out = model(word_token, char_token)

            loss, mask = loss_fn(model_out , y)
            epoch_loss_sum += float(loss)
            batch_size = word_token.size(0)
            _, predicted = torch.max(model_out, 1)
            predicted = predicted.view(batch_size, -1)
            mask = mask.view(batch_size, -1)
            correct += cal_accuracy(predicted, y, mask)

        accuracy = correct / len(datasets['eval'])
        print("Validation Accuracy = {}".format(accuracy))
        loss_history.append(epoch_loss_sum)

    # TODO: Inference on test set
    torch.save(model.state_dict(), str(args.ckpt_dir / "best.pt"))
    


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
    
# Training for TagClassifier1(RNN for character representation + Bi-GRU) and TagClassifier1(CNN + Bi-GRU)

# use_gpu = torch.cuda.is_available()
# model = TagClassifier1(embeddings=embeddings, hidden_size=1000, num_layers=3,
#                            dropout=0.1, bidirectional=True, num_class=len(tag2idx))

# learning_rate = 0.001

# def adjust_learning_rate(optimizer, epoch):
#     lr = learning_rate * (0.1 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer
# if use_gpu:
#     model = model.cuda()

# optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
# # loss_fn = torch.nn.CrossEntropyLoss()

# for epoch in tqdm(range(45)):
#     # TODO: Training loop - iterate over train dataloader and update model weights

#     optimizer = adjust_learning_rate(optimizer, epoch)
#     epoch_loss_sum = 0
#     correct = 0
#     loss_history = []
#     for i, data in enumerate(train_loader):
#         tokens = data['token']
#         tags = data['tag']
#         word_token = torch.tensor(datasets['train'].vocab.encode_batch1(tokens), dtype=torch.long)
#         char_token = torch.tensor(datasets['train'].vocab.encode_char(tokens), dtype=torch.long)
#         y = prepare_label(tags, len(word_token[0]), datasets['train'])
#         if (use_gpu):
#             word_token,char_token,y = word_token.cuda(),char_token.cuda(),y.cuda()
#         model_out = model(word_token, char_token)
#         loss, mask = loss_fn(model_out , y)
#         epoch_loss_sum += float(loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         batch_size = word_token.size(0)
#         _, predicted = torch.max(model_out, 1)
#         predicted = predicted.view(batch_size, -1)
#         mask = mask.view(batch_size, -1)
# #         print('pre:',predicted)
#         correct += cal_accuracy(predicted, y, mask)


#     accuracy = correct / len(datasets['train'])
#     print('loss:', epoch_loss_sum)
#     print("Training Accuracy = {}".format(accuracy))

#     loss_history.append(epoch_loss_sum)

#     epoch_loss_sum = 0
#     correct = 0
#     for i, data in enumerate(valid_loader):
#         tokens = data['token']
#         tags = data['tag']
#         word_token = torch.tensor(datasets['eval'].vocab.encode_batch1(tokens), dtype=torch.long)
#         char_token = torch.tensor(datasets['eval'].vocab.encode_char(tokens), dtype=torch.long)
#         y = prepare_label(tags, len(word_token[0]), datasets['eval'])
#         if (use_gpu):
#             word_token,char_token,y = word_token.cuda(),char_token.cuda(),y.cuda()

#         model_out = model(word_token, char_token)

#         loss, mask = loss_fn(model_out , y)
#         epoch_loss_sum += float(loss)
#         batch_size = word_token.size(0)
#         _, predicted = torch.max(model_out, 1)
#         predicted = predicted.view(batch_size, -1)
#         mask = mask.view(batch_size, -1)
#         correct += cal_accuracy(predicted, y, mask)

#     accuracy = correct / len(datasets['eval'])
#     print("Validation Accuracy = {}".format(accuracy))
#     loss_history.append(epoch_loss_sum)

