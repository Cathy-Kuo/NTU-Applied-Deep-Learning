from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        txts = []
        intents = []
        for sample in samples:
            txts.append(sample['text'])
            intents.append(sample['intent'])
        return {'text':txts, 'intent': intents}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
    
class TagClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        tokens = self.data[index]['tokens']
        length = len(tokens)
        token = ''
        for t in tokens:
            if t == '':
                token += ','
            else:
                token += t
            token += ' '
        token = token[:len(token)-1]
        tag = ''
        if 'tags' in self.data[index]:
            tags = self.data[index]['tags']
            for t in tags:
                tag += t
                tag += ' '
        else:
            for i in range(length):
                tag += 'O '
        tag = tag[:len(tag)-1]
        
        return {'token': token, 'tag': tag}

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

