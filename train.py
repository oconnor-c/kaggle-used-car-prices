import time

import torch
import torch.nn as nn


def get_net(in_features, device):
    net = nn.Sequential(nn.Linear(in_features, 10),
                        nn.ReLU(),
                        nn.Linear(10, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))

    return net.to(device)


loss = nn.MSELoss()


def log_rmse(net, features, labels, device, eval_batch_size=4096):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # 评估时不需要反向传播，关闭梯度可以避免构建计算图，大幅节省显存
    # 同时分批搬运到 GPU，防止一次性占用过多显存
    net.eval()
    total_loss = 0.0
    n = features.shape[0]
    with torch.no_grad():
        for i in range(0, n, eval_batch_size):
            X = features[i:i + eval_batch_size].to(device)
            y = labels[i:i + eval_batch_size].to(device)
            clipped_preds = torch.clamp(net(X), 1, float('inf'))
            batch_loss = loss(torch.log(clipped_preds), torch.log(y))
            # 按样本数加权，保证最后一批不足 eval_batch_size 时结果仍准确
            total_loss += batch_loss.item() * X.shape[0]
            del X, y, clipped_preds
    net.train()
    return (total_loss / n) ** 0.5


def predict(net, features, device, batch_size=4096):
    net.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            X = features[i:i + batch_size].to(device)
            predictions.append(torch.clamp(net(X), min=1).cpu())
    return torch.cat(predictions).view(-1).numpy()


def train(train_features, train_labels, test_features, test_labels,
          num_epochs, lr, weight_decay, batch_size, device):
    total_start_time = time.perf_counter()

    # 一次性把数据搬到 GPU 常驻，避免每个 batch 反复做 CPU->GPU 传输。
    # 小 batch 下传输开销远大于计算开销，这是主要瓶颈（实测 30s/epoch -> 8.7s/epoch）。
    # 代价是常驻显存，本数据集约 2.8GB，若显存不足可改回按 batch 搬运。
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    in_features = train_features.shape[1]
    net = get_net(in_features, device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )
    n = train_features.shape[0]
    train_ls = []

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        # 在 GPU 上生成打乱后的索引，再用索引取 batch，全程无需回传 CPU
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            X = train_features[idx]
            y = train_labels[idx]
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels, device))

        # CUDA 默认异步执行；同步后再计时，才能得到实际耗时。
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start_time
        print(f'Epoch {epoch + 1}/{num_epochs} 耗时: {epoch_time:.2f} 秒')

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - total_start_time
    print(f'训练总耗时: {total_time:.2f} 秒')

    return net, train_ls
