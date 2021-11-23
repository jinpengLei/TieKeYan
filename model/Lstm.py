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

    ratio = int(serials.shape[0] * 0.85)
    train_data = serials[:ratio]
    Max, Min = max(train_data), min(train_data)
    Scale = Max - Min
    test_data = serials[ratio:]
    # 训练集数据时间序列采样
    result = []
    for i in range(len(train_data) - window_size):
        tmp = train_data[i: i + window_size]
        label = train_data[i + window_size].reshape(1, -1)
        tmp = np.concatenate((tmp, label[0]), axis=0)
        result.append(tmp)
    result = (result - Min) / Scale
    test_sets = []

    for i in range(len(test_data) - window_size):
        tmp = test_data[i: i + window_size]
        # 后1min的数据作为label
        label = test_data[i + window_size].reshape(1, -1)
        tmp = np.concatenate((tmp, label[0]), axis=0)
        test_sets.append(tmp)
    test_sets = (test_sets - Min) / Scale
    return torch.tensor(result, dtype=torch.float).unsqueeze(2), torch.tensor(test_sets, dtype=torch.float).unsqueeze(2), Max, Min

def unnormalize(x, Max, Min):
    return x * (Max - Min) + Min

def train(epoch, train_sets, test_sets, model, optimizer, window_size, criterion, Max, Min):
    train_loss_list = []
    test_loss_list = []
    train_sets = train_sets.permute(1, 0, 2)
    test_sets = test_sets.permute(1, 0, 2)
    for i in range(epoch):
        losses = 0
        print(train_sets.shape)
        x = train_sets[:window_size, :, :].type(torch.float)
        label = train_sets[-1, :, :].type(torch.float)
        pred = model(x)
        pred = pred.squeeze(0)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())

        print("epoch: {},loss : {}".format(i + 1, loss))
        # 测试集上的loss
        x = test_sets[:window_size, :, :].type(torch.float)
        label = test_sets[-1, :, :].type(torch.float)
        pred = model(x)
        pred = pred.squeeze(0)
        loss = criterion(pred, label)
        test_loss_list.append(loss.item())
    return train_loss_list, test_loss_list


def test(model, test_sets, window_size, criterion, Max, Min):
    model.eval()
    test_sets = test_sets.permute(1, 0, 2)
    x = test_sets[:window_size, :, :].type(torch.float)
    label = test_sets[-1, :, :].type(torch.float)
    pred = model(x)
    pred = pred.squeeze(0)
    test_loss = criterion(unnormalize(pred, Max, Min), unnormalize(label, Max, Min))
    print("test loss: {}".format(test_loss))
    return pred, label


def show_preds(test_preds, test_label, Max, Min):

    x = np.arange(1, test_preds.shape[0] + 1)
    test_preds = unnormalize(test_preds, Max, Min)
    test_label = unnormalize(test_label, Max, Min)
    plt.plot(x, test_preds.detach().numpy(), label='predict')
    plt.plot(x, test_label.detach().numpy(), label="original")
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


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

if __name__ == "__main__":
    criterion = RMSELoss()
    window_size = 8
    epoch = 1000
    lr = 0.05
    model = LSTMModel(1, 32, 1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_sets, test_sets, Max, Min = train_test_split(window_size)

    train_loss_list, test_loss_list = train(epoch, train_sets, test_sets, model, optimizer, window_size, criterion, Max, Min)

    test_preds, test_label = test(model, test_sets, window_size, criterion, Max, Min)

    show_preds(test_preds, test_label, Max, Min)

    epoches = np.arange(1, epoch + 1)
    train_loss_list = np.array(train_loss_list)
    test_loss_list = np.array(test_loss_list)

    plt.plot(epoches, train_loss_list, label="train")
    plt.plot(epoches, test_loss_list, label="test")
    plt.legend()
    plt.plot()
    plt.show()