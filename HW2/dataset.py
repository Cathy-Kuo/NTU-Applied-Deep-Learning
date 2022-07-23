from torch.utils.data import Dataset
import torch
 
    
class MultiChoiceDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, data, tokenizer):
        self.data = data
        self.len = len(self.data)
        self.tokenizer = tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        ##注意test no label
        qs = self.data[idx][0]
        ps = self.data[idx][1]
        label = self.data[idx][2]
        
#roberta 510
        all_ = self.tokenizer(text=qs, text_pair=ps, add_special_tokens=True,max_length=512,truncation=True, return_tensors='pt', padding=True)
        tokens_tensor = torch.tensor(all_['input_ids'])
        segments_tensor = torch.tensor(all_['token_type_ids'])
        mask_tensor = torch.tensor(all_['attention_mask'])
        label_tensor = torch.tensor(label)
        
        return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)
#        return (tokens_tensor, mask_tensor, label_tensor)
    
    def __len__(self):
        return self.len
    

def add_token_positions(tokenizer, encoding, start_idx, end_idx):
    start_position = encoding.char_to_token(start_idx, sequence_index=1)
    end_position = encoding.char_to_token(end_idx-1, sequence_index=1)

    # if start position is None, the answer passage has been truncated
    if start_position is None:
        start_position = 512
    # end position cannot be found, char_to_token found space, so shift position until found
    shift = 1
    while end_position is None:
        end_position = encoding.char_to_token(end_idx - shift)
        shift += 1
    if start_position is not None:
        start_position = torch.tensor(start_position)
    if end_position is not None:
        end_position = torch.tensor(end_position)
    return start_position, end_position
    
class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.len = len(self.data)
        self.tokenizer = tokenizer 
    
    def __getitem__(self, idx):
        #注意test no label
        text_q = self.data[idx][0]
        text_p = self.data[idx][1]
        ans = self.data[idx][2]
        
        start_idx = ans['start']
        end_idx = ans['end']
        
        all_ = self.tokenizer(text=text_q, text_pair=text_p,add_special_tokens=True,max_length=512,truncation=True)
        tokens_tensor = torch.tensor(all_['input_ids'])
        segments_tensor = torch.tensor(all_['token_type_ids'])
        mask_tensor = torch.tensor(all_['attention_mask'])
        
        start_tensor, end_tensor = add_token_positions(self.tokenizer, all_, start_idx, end_idx)
#        print(ans['text'],self.tokenizer.convert_ids_to_tokens(tokens_tensor[start_tensor:end_tensor+1]))
        
        return (tokens_tensor, segments_tensor, mask_tensor, start_tensor, end_tensor, all_)
#        return (tokens_tensor, mask_tensor, start_tensor, end_tensor)
    
    def __len__(self):
        return self.len
    
class QADataset1(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)
    
    def __getitem__(self, idx):
        
        
        return self.data[idx]
    
    def __len__(self):
        return self.len
        

    


    