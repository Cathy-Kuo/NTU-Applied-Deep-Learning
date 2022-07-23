from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn.functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(self,embeddings: torch.tensor,hidden_size: int,num_layers: int,dropout: float,bidirectional: bool,num_class: int):
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.direction = 1
        if bidirectional:
            self.direction = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size=embeddings.size(1), 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional,batch_first = True)
    
        self.L1 = torch.nn.Linear(2*self.direction*hidden_size, num_class)


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch_size = batch.size(0)
        len_txt = batch.size(1)
        embeds = self.embed(batch)
        gru_out, h = self.gru(embeds)
        out = torch.cat((gru_out[:,0],gru_out[:,-1]),1)
        out = self.L1(out)
        return out
    
    
class TagClassifier(torch.nn.Module):
    def __init__(self,embeddings: torch.tensor,hidden_size: int,num_layers: int,dropout: float,bidirectional: bool,num_class: int):
        super(TagClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.direction = 1
        if bidirectional:
            self.direction = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size=embeddings.size(1), 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional,batch_first=True)
    
        self.L1 = torch.nn.Linear(self.direction*hidden_size, num_class)
        

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch_size = batch.size(0)
        len_txt = batch.size(1)
        embeds = self.embed(batch)
        gru_out, _ = self.gru(embeds)
        out = gru_out.reshape(-1, gru_out.shape[2])
        out = self.L1(out)
        out = F.log_softmax(out, dim=1)
        return out

    
class TagClassifier1(torch.nn.Module):
    def __init__(self,embeddings: torch.tensor,hidden_size: int,num_layers: int,dropout: float,bidirectional: bool,num_class: int):
        super(TagClassifier1, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.direction = 1
        if bidirectional:
            self.direction = 2
        self.hidden_size = hidden_size
        self.char_hidden = 200
        self.num_layers = num_layers
        self.embed_dim = embeddings.size(1)
        self.char_rnn = torch.nn.GRU(input_size=self.embed_dim, 
                           hidden_size=self.char_hidden, 
                           num_layers=3, dropout=0.3,
                           bidirectional=True,batch_first=True)
        
        self.gru = torch.nn.GRU(input_size=self.embed_dim + (self.char_hidden*2*15), 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional,batch_first=True)
        
        self.drop_layer = torch.nn.Dropout(p=0.2)
    
        self.L1 = torch.nn.Linear(self.direction*hidden_size, num_class)


    def forward(self, word_token, char_token) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch_size = char_token.size(0)
        word_len = char_token.size(1)
        char_len = char_token.size(2)
        
        char_embed = self.embed(char_token)
        word_embed = self.embed(word_token)
        char_enc, _ = self.char_rnn(char_embed.view(batch_size,-1,self.embed_dim))
        char_enc = char_enc.reshape(batch_size, word_len, -1)
        
        embeds = torch.cat([word_embed, char_enc], dim=-1)
        
        embeds = self.drop_layer(embeds)
        
        gru_out, _ = self.gru(embeds)
        out = gru_out.reshape(-1, gru_out.shape[2])
        out = self.L1(out)
        out = F.log_softmax(out, dim=1)
        return out
    
    
class CharCNN(torch.nn.Module):
    def __init__(self,
                 embeddings,
                 max_word_len,
                 num_filters,
                 final_char_dim):
        super(CharCNN, self).__init__()

        self.char_emb = Embedding.from_pretrained(embeddings, freeze=False)
        self.embed_dim = embeddings.size(1)

        kernel_lst = [2, 3, 4, 5]

        self.convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(self.embed_dim, num_filters, kernel_size, padding=kernel_size // 2),
                torch.nn.ReLU(), 
                torch.nn.MaxPool1d(max_word_len - kernel_size + 1)
            ) for kernel_size in kernel_lst
        ])

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(num_filters * len(kernel_lst), 500),
            torch.nn.ReLU(), 
            torch.nn.Dropout(0.1),
            torch.nn.Linear(500, final_char_dim)
        )

    def forward(self, x):
        """
        :param x: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, final_char_dim)
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_emb(x)
        x = x.view(batch_size * max_seq_len, max_word_len, -1)
        x = x.transpose(2, 1) 
        conv_lst = [conv(x) for conv in self.convs]
        conv_concat = torch.cat(conv_lst, dim=-1) 
        conv_concat = conv_concat.view(conv_concat.size(0), -1) 

        output = self.linear(conv_concat)
        output = output.view(batch_size, max_seq_len, -1) 
        return output
    
    
class TagClassifier2(torch.nn.Module):
    def __init__(self,embeddings: torch.tensor,hidden_size: int,num_layers: int,dropout: float,bidirectional: bool,num_class: int):
        super(TagClassifier2, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.direction = 1
        if bidirectional:
            self.direction = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embeddings.size(1)
        final_char_dim = 300
        self.char_cnn = CharCNN(embeddings, max_word_len=15,
                                num_filters=256,
                                final_char_dim=final_char_dim)
        
        self.gru = torch.nn.GRU(input_size=self.embed_dim + final_char_dim, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional,batch_first=True)
    
        self.L1 = torch.nn.Linear(self.direction*hidden_size, num_class)


    def forward(self, word_token, char_token) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        word_embed = self.embed(word_token)
        char_embed = self.char_cnn(char_token)

        w_c_emb = torch.cat([word_embed, char_embed], dim=-1)

        gru_out, _ = self.gru(w_c_emb)
        out = gru_out.reshape(-1, gru_out.shape[2])
        out = self.L1(out)
        out = F.log_softmax(out, dim=1)
        return out
    
