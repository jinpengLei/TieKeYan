from torch import  nn
from utils import train_test_split, train, test,show_preds
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

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
    window_size = 20
    epoch = 50
    lr = 0.1
    model = LSTMModel(4, 256, 4)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_loader, test_loader ,E_scaler, N_scaler, V_scaler, C_scaler = train_test_split(window_size)

    train_loss_list, test_loss_list = train(epoch,train_loader,test_loader, model,optimizer,window_size,criterion, "torch")

    test_preds, test_label = test(model,test_loader,window_size,criterion,"torch")

    show_preds(15,test_preds,test_label, V_scaler, 2 ,0)

    train_loss_list = np.array(train_loss_list)


    test_loss_list = np.array(test_loss_list)
    epoches = np.arange(1, epoch + 1)
    plt.plot(epoches, train_loss_list, label = "train")
    plt.plot(epoches, test_loss_list, label = "test")
    plt.legend()
    plt.plot()
    plt.show()

