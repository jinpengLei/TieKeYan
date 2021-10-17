import numpy as np
import pandas as pd
import time
TIMEFORMAT = "%Y-%m-%d %H:%M:%S"
#数据统计处理
def convert_time(timestamp):
    loca_time = time.localtime(int(timestamp))
    return time.strftime(TIMEFORMAT, loca_time)
def readtxt(filename):
    cou = 0
    with open(filename) as f:
        line = f.readline()
        time_list = []
        fault_code_list = []
        while line:
            line = line[4:]
            per_line_list = line.split('|')
            data_nums = len(per_line_list)
            for i in range(7, data_nums):
                fault = per_line_list[i]
                data_info_list = fault.split(',')
                time_list.append(data_info_list[3])
                fault_code_list.append(data_info_list[1])
            cou = cou + 1
            line = f.readline()
    dataframe = pd.DataFrame({'time': time_list, 'fault_code': fault_code_list})
    dataframe = dataframe.sort_values(by='time')
    dataframe['time'] = dataframe['time'].apply(lambda x: convert_time(x))
    return dataframe

filename = "../data/fault.txt"
data = readtxt(filename)
print(data)
data.to_csv("../data/test.csv",index=False,sep=',')