from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd

def train_test_split(window_size):
    df = pd.read_csv("../utils/multi_series.csv")
    df = df[['E', 'N', 'lcsd', 'xfwd']]
    print(df)
    data1 = df.values
    print(data1)
    print(type(data1))
    print(data1.shape)
    data1 = np.expand_dims(data1, axis=1)
    print(data1.shape)
    scaler_data = np.zeros((data1.shape[0], data1.shape[1], data1.shape[2]))

    E_scaler = MinMaxScaler()  # 经度归一化
    N_scaler = MinMaxScaler()  # 纬度归一化
    V_scaler = MinMaxScaler()  # 列车速度归一化
    C_scaler = MinMaxScaler()  # 新风温度归一化

    scaler_data[:, :, 0] = E_scaler.fit_transform(data1[:, :, 0])  # 经度
    scaler_data[:, :, 1] = N_scaler.fit_transform(data1[:, :, 1])  # 纬度
    scaler_data[:, :, 2] = V_scaler.fit_transform(data1[:, :, 2])  # 列车速度
    scaler_data[:, :, 3] = C_scaler.fit_transform(data1[:, :, 3])  # 列车速度

    ratio = int(data1.shape[0] * 0.75)
    train_data = scaler_data[:ratio, :, :]
    test_data = scaler_data[ratio:, :, :]

    # 训练集数据时间序列采样
    result = []
    for i in range(len(train_data) - window_size - 1):
        tmp = train_data[i: i + window_size, :, :]
        tmp = tmp.reshape(-1, 4)
        # 后1min的数据作为label
        label = train_data[i + window_size + 1, :, :].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        result.append(tmp)

    train_loader = DataLoader(result, batch_size=30, shuffle=False)

    test_sets = []

    for i in range(len(test_data) - window_size - 1):
        tmp = test_data[i: i + window_size, :, :]
        tmp = tmp.reshape(-1, 4)
        # 后1min的数据作为label
        label = test_data[i + window_size + 1, :, :].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        test_sets.append(tmp)
    test_loader = DataLoader(test_sets, batch_size=36, shuffle=False)
    # 返回MinMaxScaler以便反归一化
    return train_loader, test_loader, E_scaler, N_scaler, V_scaler, C_scaler

def train(epoch, train_loader, test_loader, model, optimizer, window_size, criterion,alt):
    train_loss_list = []
    test_loss_list = []
    for i in range(epoch):
        losses = 0
        for batch in train_loader:
            # [src_len, batch, embedded]
            batch = batch.permute(1, 0, 2)
            x = batch[:window_size, :, :].type(torch.float)
            label = batch[-1, :, :].type(torch.float)
            if alt == "my":
                pred, _ = model(x)
                pred = pred[-1, :, :]
            else:
                pred = model(x)
                pred = pred.squeeze(0)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        train_loss_list.append(losses)
        
        print("epoch: {},loss : {}".format(i + 1, losses))
        #测试集上的loss
        losses = 0
        for batch in test_loader:
            batch = batch.permute(1, 0, 2)
            x = batch[:window_size, :, :].type(torch.float)
            label = batch[-1, :, :].type(torch.float)
            if alt == "my":
                pred, _ = model(x)
                pred = pred[-1, :, :]
            else:
                pred = model(x)
                pred = pred.squeeze(0)
            loss = criterion(pred, label)
            losses += loss.item()
        test_loss_list.append(losses)
        
    return train_loss_list, test_loss_list

def test(model, test_loader,window_size,criterion, alt):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_label = []
    for batch in test_loader:
        batch = batch.permute(1, 0, 2)
        x = batch[:window_size, :, :].type(torch.float)
        label = batch[-1, :, :].type(torch.float)
        if alt == "my":
            pred, _ = model(x)
            pred = pred[-1, :, :]
        else:
            pred = model(x)
            pred = pred.squeeze(0)
        test_preds.append(pred)
        test_label.append(label)
        loss = criterion(pred, label)
        test_loss += loss.item()
    print("test loss: {}".format(test_loss))
    return test_preds, test_label

def show_preds(batch, test_preds, test_label, scaler, type, no):
    # 获取前batch的预测数据
    p = torch.stack(test_preds[:batch], dim=0).reshape(-1, 1, 4)
    t = torch.stack(test_label[:batch], dim=0).reshape(-1, 1, 4)

    predict = scaler.inverse_transform(p[:, :, type].detach().numpy())
    labels = scaler.inverse_transform(t[:, :, type].detach().numpy())
    no_predict = predict[:, no]
    no_labels = labels[:, no]

    x = np.arange(1, p.shape[0] + 1)

    plt.plot(x, no_predict, label='predict')
    plt.plot(x, no_labels, label="original")
    plt.legend()
    plt.show()
