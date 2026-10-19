import pandas as pd
import torch
def preprocess_data(train_data,test_data):
    all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]),axis=0)
    
    #对数值特征进行标准化
    numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
    all_features[numeric_features]=all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
    all_features[numeric_features]=all_features[numeric_features].fillna(0)
    
    #对类别特征进行独热编码，也就是文本类型，不是数值类型的特征
    all_features=pd.get_dummies(all_features,dummy_na=True)
    all_features=all_features.astype(float)
    
    n_train=train_data.shape[0]
    train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float32)
    test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float32)
    train_labels=torch.tensor(train_data.iloc[:,-1].values,dtype=torch.float32).view(-1,1)
    
    return train_features,test_features,train_labels