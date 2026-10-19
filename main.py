# Windows 下必须先导入 torch，再导入 pandas/numpy，否则 torch 的 DLL 初始化会失败（WinError 1114）
import torch
import pandas as pd
from preprocess import preprocess_data
from sha1_check import sha1_check
from train import predict, train
from gpu_monitor import GPUMonitor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_HUB=dict()
DATA_HUB['used_car_prices_train']=('used_car_prices/train.csv','b9ebb482a7f97ecaec17cb7fb347ac78ba1beca1')
DATA_HUB['used_car_prices_test']=('used_car_prices/test.csv','392a215375f587c4dcc297d2faf5535d90dcae1c')

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}', end='')
    if device.type=='cuda':
        print(f' ({torch.cuda.get_device_name(0)})')
    else:
        print(' (未检测到可用 GPU，将使用 CPU 训练)')
    
    for filename in DATA_HUB:
        sha1_check(filename,DATA_HUB)
    
    train_data=pd.read_csv(DATA_HUB['used_car_prices_train'][0])
    test_data=pd.read_csv(DATA_HUB['used_car_prices_test'][0])
    
    train_features,test_features,train_labels=preprocess_data(train_data,test_data)
    test_labels=None
    
    gpu_monitor = GPUMonitor(interval=1.0, output_path='gpu_usage.png')
    gpu_monitor.start()
    try:
        net, train_ls = train(
            train_features, train_labels, test_features, test_labels,
            num_epochs=100, lr=0.01, weight_decay=0, batch_size=1024,
            device=device
        )
    finally:
        gpu_monitor.stop()
        print(train_ls)
    submission = pd.read_csv('used_car_prices/sample_submission.csv')
    submission['price'] = predict(net, test_features, device)
    if len(submission) != len(test_data):
        raise ValueError('submission.csv and test.csv have different row counts')
    if not submission['id'].equals(test_data['id']):
        raise ValueError('submission.csv IDs do not match test.csv IDs')
    submission.to_csv('used_car_prices/submission.csv', index=False)
    print('提交文件已生成: used_car_prices/submission.csv')
    