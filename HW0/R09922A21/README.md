ADL HW0
首先將三個csv 檔利用dataframe 讀進來後，將dev 及train concat 在一起形成training data。
接著使用training data 的text 去fit TfidfVectorizer，並將training 及testing data的 text 做transform。
將training data 建成Dataloader，batch_size = 32，shuffle = True。
Network 的部分則建立8個Layer，Active function 使用ReLu，Loss 使用CrossEntropyLoss，Optimizer 使用Adam，training epoch = 2。
最後將transform 後的testing data 餵給Model 後即完成這次的HW0。

