from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
read_file_name = '../data/counts_serial.csv'
def train_test_split(window_size):
    data =  pd.read_csv(read_file_name)
    serials = data['fault_nums']
    serials = serials.values
    print(serials)
    scaler = MinMaxScaler()  # 归一化
    serials = serials.reshape(-1, 1)
    print(serials.shape[0])
    scaler_data = scaler.fit_transform(serials)

    ratio = int(serials.shape[0] * 0.85)
    train_data = scaler_data[:ratio]
    test_data = scaler_data[ratio:]
    # 训练集数据时间序列采样
    result = []
    for i in range(len(train_data) - window_size - 1):
        tmp = train_data[i: i + window_size]
        label = train_data[i + window_size + 1].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        result.append(tmp)

    train_loader = DataLoader(result, batch_size=200, shuffle=False)

    test_sets = []

    for i in range(len(test_data) - window_size - 1):
        tmp = test_data[i: i + window_size]
        # 后1min的数据作为label
        label = test_data[i + window_size + 1].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        test_sets.append(tmp)
    test_loader = DataLoader(test_sets, batch_size=19, shuffle=False)
    # 返回MinMaxScaler以便反归一化
    return train_loader, test_loader, scaler


def train(epoch, train_loader, test_loader, model, optimizer, window_size, criterion, alt):
    train_loss_list = []
    test_loss_list = []
    for i in range(epoch):
        losses = 0
        for batch in train_loader:
            print(batch.shape)
            batch = batch.permute(1, 0, 2)
            print(batch.shape)
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
        # 测试集上的loss
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


def test(model, test_loader, window_size, criterion, alt):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_label = []
    for batch in test_loader:
        batch = batch.permute(1, 0, 2)
        x = batch[:window_size, :, :].type(torch.float)
        print(x)
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
    p = torch.stack(test_preds[:batch], dim=0)
    t = torch.stack(test_label[:batch], dim=0)

    predict = scaler.inverse_transform(p[:, :, type].detach().numpy())
    labels = scaler.inverse_transform(t[:, :, type].detach().numpy())
    no_predict = predict[0, :]
    no_labels = labels[0, :]

    x = np.arange(1, p.shape[1] + 1)
    print(no_predict)
    plt.plot(x, no_predict, label='predict')
    plt.plot(x, no_labels, label="original")
    plt.legend()
    plt.show()


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
            x: [src_len, batch_size, embedding]
        '''
        _, (hid, cell) = self.lstm(x)

        output = self.out(hid)

        return output

if __name__ == "__main__":
    criterion = nn.MSELoss()
    window_size = 5
    epoch = 10
    lr = 0.1
    model = LSTMModel(1, 32, 1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_loader, test_loader, scaler = train_test_split(window_size)

    train_loss_list, test_loss_list = train(epoch, train_loader, test_loader, model, optimizer, window_size, criterion,
                                            "torch")

    test_preds, test_label = test(model, test_loader, window_size, criterion, "torch")

    show_preds(15, test_preds, test_label, scaler, 0, 1)

    epoches = np.arange(1, epoch + 1)
    train_loss_list = np.array(train_loss_list)
    test_loss_list = np.array(test_loss_list)

    plt.plot(epoches, train_loss_list, label="train")
    plt.plot(epoches, test_loss_list, label="test")
    plt.legend()
    plt.plot()