#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch

train = pd.read_csv("train.csv")
dev = pd.read_csv("dev.csv")
test = pd.read_csv("test.csv")
train.head()

train_df = pd.concat([train,dev], ignore_index=True)


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

tfIdfVectorizer_1000 = TfidfVectorizer(max_features = 10000, tokenizer=nltk.word_tokenize)

tfIdfVectorizer_1000.fit(train_df['text'])

X_train = tfIdfVectorizer_1000.transform(train_df['text'])
y_train = train_df['Category']

X_test = tfIdfVectorizer_1000.transform(test['text'])
y_test = test['Category']

print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)


from torch.utils.data import TensorDataset, DataLoader

X_train = X_train.todense()

X_train = torch.tensor(X_train, dtype = torch.float)
y_train = torch.tensor(y_train)

train_data = TensorDataset(X_train, y_train)

dataLoader = DataLoader(train_data, batch_size=32, shuffle=True)


from tqdm.notebook import tqdm

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.L1 = nn.Linear(10000,4096)
        self.L2 = nn.Linear(4096,2048)
        self.L3 = nn.Linear(2048,1024)
        self.L4 = nn.Linear(1024,512)
        self.L5 = nn.Linear(512,256)
        self.L6 = nn.Linear(256,128)
        self.L7 = nn.Linear(128,64)
        self.L8 = nn.Linear(64,32)
        self.output = nn.Linear(32,2)
    def forward(self , x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))
        x = F.relu(self.L5(x))
        x = F.relu(self.L6(x))
        x = F.relu(self.L7(x))
        x = F.relu(self.L8(x))
        x = self.output(x)
        return x


import torch
use_gpu = torch.cuda.is_available()
print(use_gpu)

model = network()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())
loss_history = []
batch_size = 32

X_test = tfIdfVectorizer_1000.transform(test['text'])
X_test = X_test.todense()
X_test = torch.tensor(X_test, dtype = torch.float)
X_test = X_test.cuda()

if(use_gpu):
    model = model.cuda()
    
for e in tqdm(range(2)):
    epoch_loss_sum = 0
    correct = 0
    for x , y in tqdm(dataLoader):
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
        
    accuracy = 100 * correct / len(X_train)
    print("Accuracy = {}".format(accuracy))
    
    loss_history.append(epoch_loss_sum)


import numpy as np

pred_result = model(X_test)

_, predicted = torch.max(pred_result, 1)

print(predicted[:5])


test['Category'] = predicted.cpu().detach().numpy()

test = test.drop(['text'], axis=1)
test.head()
test.to_csv('submission.csv',index=0)
