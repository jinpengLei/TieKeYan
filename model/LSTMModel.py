import torch
from torch import nn
from utils import train_test_split, train, test, show_preds, data_prepare, new_train, get_device
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        '''
            x: [src_len, batch_size, embedding]
        '''
        x = x.to(torch.float32)
        o, (hid, cell) = self.lstm(x)
        # print("Shape")
        # print(o.shape)
        # print(hid.shape)
        # print(cell.shape)
        output = self.out(hid.reshape((-1, hid.shape[-1])))
        
        return output


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

if __name__ == "__main__":
    # criterion = nn.MSELoss()
    # criterion = RMSELoss()
    device = get_device()
    criterion = nn.MSELoss()
    window_size = 96
    epochs = 100
    lr = 0.0001
    model = LSTMModel(12, 256, 12)
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set, train_loader, vali_set, vali_data, test_set, test_data = data_prepare(root_path="../dataset", data_path="WTH.csv", window_size=window_size)
    train_loss_list, test_loss_list = new_train(epochs, train_set, train_loader, vali_set, vali_data, test_set, test_data, model, optimizer, criterion, device)
    # train_loader, test_loader ,E_scaler, N_scaler, V_scaler, C_scaler = train_test_split(window_size)

    # train_loss_list, test_loss_list = train(epoch,train_loader,test_loader, model,optimizer,window_size,criterion, "torch")
    #
    # test_preds, test_label = test(model,test_loader,window_size,criterion,"torch")
    # scaler_list = [E_scaler, N_scaler, V_scaler, C_scaler]
    # for i in range(4):
    #     show_preds(15, test_preds, test_label, scaler_list[i], i, 0)

    train_loss_list = np.array(train_loss_list)


    test_loss_list = np.array(test_loss_list)
    epoches = np.arange(1, epochs + 1)
    plt.plot(epoches, train_loss_list, label = "train")
    plt.plot(epoches, test_loss_list, label = "test")
    font = {"size": 12}
    plt.xlabel("训练轮数", font)
    plt.ylabel("MSE", font)
    plt.legend()
    plt.plot()
    plt.show()


