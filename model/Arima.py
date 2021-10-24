import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf
read_file_name = '../data/counts_serial.csv'
class ArimaModel():
    def __init__(self, filename):
        self.serial = pd.read_csv(filename)
        print(self.serial)

    def draw_pacf(self):
        plot_acf(self.serial['fault_nums'])
        plt.show()

if __name__ == "__main__":
    arimamodel = ArimaModel(read_file_name)
    arimamodel.draw_pacf()