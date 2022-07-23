from torch.utils.data import Dataset
import torch
 
    

class NLGDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        text = self.data[idx]['maintext']
        label = self.data[idx]['title']
        
        text_all = self.tokenizer(text, max_length=712, truncation=True, padding=True, pad_to_max_length=True, return_tensors='pt')
        with self.tokenizer.as_target_tokenizer():
            label_all = self.tokenizer(label, max_length=120, truncation=True, padding=True, pad_to_max_length=True, return_tensors='pt')

        text_id = torch.tensor(text_all['input_ids'].squeeze())
        text_mask = torch.tensor(text_all['attention_mask'].squeeze())
        label_id = torch.tensor(label_all['input_ids'].squeeze())
        label_mask = torch.tensor(label_all['attention_mask'].squeeze())
        
        return (text_id, text_mask, label_id, label_mask)
    
class NLGDataset1(Dataset):

    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        text = self.data[idx]['maintext']
        
        text_all = self.tokenizer(text, max_length=712, truncation=True, padding=True, pad_to_max_length=True, return_tensors='pt')

        text_id = torch.tensor(text_all['input_ids'].squeeze())
        text_mask = torch.tensor(text_all['attention_mask'].squeeze())
        
        return (text_id, text_mask)