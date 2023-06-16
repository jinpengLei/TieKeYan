from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
from torch import nn
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os
import time

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class MyDataset(Dataset):
    def __init__(self, root_path='/dataset', flag="train", data_path="WTH.csv", scale=True, window_size=20):
        assert flag in ['train', 'test', 'vali']
        type_map = {'train':0 , 'vali':1, 'test':2}
        self.set_type= type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.window_size = window_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        timefeature = cols[0]
        cols.remove(timefeature)
        df_data = df_raw[cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.window_size, len(df_raw) - num_test - self.window_size]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.window_size

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[s_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.window_size

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_device(gpu=0, use_multi_gpu=False, devices="0,1"):
    use_gpu = True if torch.cuda.is_available()  else False
    if use_gpu and use_multi_gpu:
        dvices = devices.replace(' ', '')
        device_ids = devices.split(',')
        device_ids = [int(id_) for id_ in device_ids]
        gpu = device_ids[0]
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) if not use_multi_gpu else devices
        device = torch.device('cuda:{}'.format(gpu))
        print('Use GPU: cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device
def data_provider(root_path, data_path, window_size, batch_size, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = batch_size

    data_set = MyDataset(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        window_size=window_size
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=8,
        drop_last=drop_last)
    return data_set, data_loader


def data_prepare(root_path, data_path, window_size, batch_size=32):
    train_set, train_loader = data_provider(root_path, data_path, window_size, batch_size, flag="train")
    vali_set, vali_loader = data_provider(root_path, data_path, window_size, batch_size, flag="vali")
    test_set, test_loader = data_provider(root_path, data_path, window_size, batch_size, flag="test")
    return train_set, train_loader, vali_set, vali_loader, test_set, test_loader


def train_test_split(window_size):
    df = pd.read_csv("../utils/multi_series.csv")
    df = df[['E', 'N', 'lcsd', 'xfwd']]
    # print(df)
    # print("df shape")
    # print(df.shape)
    data1 = df.values
    # print(data1)
    # print(type(data1))
    # print(data1.shape)
    data1 = np.expand_dims(data1, axis=1)
    # print(data1.shape)
    scaler_data = np.zeros((data1.shape[0], data1.shape[1], data1.shape[2]))

    E_scaler = MinMaxScaler()  # 经度归一化
    N_scaler = MinMaxScaler()  # 纬度归一化
    V_scaler = MinMaxScaler()  # 列车速度归一化
    C_scaler = MinMaxScaler()  # 新风温度归一化

    scaler_data[:, :, 0] = E_scaler.fit_transform(data1[:, :, 0])  # 经度
    scaler_data[:, :, 1] = N_scaler.fit_transform(data1[:, :, 1])  # 纬度
    scaler_data[:, :, 2] = V_scaler.fit_transform(data1[:, :, 2])  # 列车速度
    scaler_data[:, :, 3] = C_scaler.fit_transform(data1[:, :, 3])  # 新风温度

    ratio = int(data1.shape[0] * 0.75)
    train_data = scaler_data[:ratio, :, :]
    test_data = scaler_data[ratio:, :, :]
    print(len(train_data))

    # 训练集数据时间序列采样
    result = []
    for i in range(len(train_data) - window_size - 1):
        tmp = train_data[i: i + window_size, :, :]
        tmp = tmp.reshape(-1, 4)
        # 后1min的数据作为label
        label = train_data[i + window_size + 1, :, :].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        result.append(tmp)
    print(result[0].shape)
    train_loader = DataLoader(result, batch_size=32, shuffle=False)

    test_sets = []

    for i in range(len(test_data) - window_size - 1):
        tmp = test_data[i: i + window_size, :, :]
        tmp = tmp.reshape(-1, 4)
        # 后1min的数据作为label
        label = test_data[i + window_size + 1, :, :].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        test_sets.append(tmp)
    test_loader = DataLoader(test_sets, batch_size=32, shuffle=False)
    # 返回MinMaxScaler以便反归一化
    return train_loader, test_loader, E_scaler, N_scaler, V_scaler, C_scaler


def vali(vali_loader, model, criterion, device):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            outputs = model(batch_x)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)

            total_loss.append(loss)

        total_loss = np.average(total_loss)
        model.train()
        return total_loss
def train(epoch, train_loader, test_loader, model, optimizer, window_size, criterion,alt):
    train_loss_list = []
    test_loss_list = []
    train_dataset = torch.tensor(np.array(train_loader.dataset))
    test_dataset = torch.tensor(np.array(test_loader.dataset))
    X = train_dataset[:, :window_size, :].type(torch.float)
    Y = train_dataset[:, -1, :].type(torch.float)
    X_test = test_dataset[:, :window_size, :].type(torch.float)
    Y_test = test_dataset[:, -1, :].type(torch.float)
    for i in range(epoch):
        losses = 0
        train_loss = []
        for batch in train_loader:
            # [src_len, batch, embedded]
            x = batch[:, :window_size, :].type(torch.float)
            label = batch[:, -1, :].type(torch.float)
            # print(x.shape)
            if alt == "my":
                pred, _ = model(x)
                pred = pred[:, -1, :]
            else:
                pred = model(x)
                pred = pred.squeeze(0)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss_val = np.average(train_loss)
        train_loss_list.append(train_loss_val)
        print("epoch: {},loss : {}".format(i + 1, train_loss_val))
        test_loss_list.append(criterion(model(X_test), Y_test).item())
    mae_loss_f = nn.L1Loss()
    mae_loss = mae_loss_f(model(X_test), Y_test)
    print("mse loss: {}".format(mae_loss))
        
    return train_loss_list, test_loss_list



def new_train(epochs, train_set, train_loader, vali_set, vali_loader, test_set, test_loader, model, optimizer, criterion, device):

    time_now = time.time()
    train_steps = len(train_loader)
    train_loss_list = []
    test_loss_list = []
    for epoch in range(epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((epochs - epoch) * train_steps - i)
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward
            optimizer.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(vali_loader, model, criterion, device)
        test_loss = vali(test_loader, model, criterion, device)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))

    return train_loss_list, test_loss_list



def test(model, test_loader,window_size,criterion, alt):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_label = []
    for batch in test_loader:
        x = batch[:, :window_size, :].type(torch.float)
        label = batch[:, -1, :].type(torch.float)
        if alt == "my":
            pred, _ = model(x)
            pred = pred[:, -1, :]
        else:
            pred = model(x)
            pred = pred.squeeze(0)
        test_preds.append(pred)
        test_label.append(label)
        loss = criterion(pred, label)
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
    font = {"size" : 12}
    yname = {0 : "经度", 1:"纬度", 2: "列车速度", 3:"新风温度"}
    plt.plot(x, no_predict, label='预测值')
    plt.plot(x, no_labels, label=yname[type])
    plt.xlabel("时间片", font)
    plt.ylabel(yname[type], font)
    plt.legend()
    plt.show()
