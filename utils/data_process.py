import numpy as np
import pandas as pd
import time

#数据统计处理
origin_data_filename = "../data/fault.txt"
class DataProcess():
    TIMEFORMAT = "%Y-%m-%d %H:%M:%S"
    DATAITEM = ["time", "car_code", "fault_code", "is_master", "fault_pattern", "fault_state"]
    DATAITEMINDEX = {"is_master": 0, "fault_code": 1, "car_code": 2, "time": 3, "fault_pattern": 4, "fault_state": 5}
    def __init__(self, filename = origin_data_filename, data_item = DATAITEM):
        self.origin_data = self.readtxt(filename, data_item)
    def convert_time(self, timestamp):
        loca_time = time.localtime(int(timestamp))
        return time.strftime(DataProcess.TIMEFORMAT, loca_time)

    def readtxt(self, filename, data_item):
        cou = 0
        with open(filename) as f:
            line = f.readline()
            data_item_nums = len(data_item)
            data_list = [[] for _ in range(data_item_nums)]
            while line:
                line = line[4:]
                per_line_list = line.split('|')
                data_nums = len(per_line_list)
                for i in range(7, data_nums):
                    fault = per_line_list[i]
                    data_info_list = fault.split(',')
                    for j in range(data_item_nums):
                        index = DataProcess.DATAITEMINDEX[data_item[j]]
                        data_list[j].append(data_info_list[index].strip())
                cou = cou + 1
                line = f.readline()
        print(cou)
        csv_content = {}
        for i in range(data_item_nums):
            csv_content[data_item[i]] = data_list[i]
        dataframe = pd.DataFrame(csv_content)
        dataframe = dataframe.sort_values(by='time')
        dataframe['time'] = dataframe['time'].apply(lambda x: self.convert_time(x))
        return dataframe

    def to_csv(self, target_file_name = '../data/test.csv'):
        self.origin_data.to_csv(target_file_name, index=False, sep=',')

if __name__ == "__main__":
    data_item_list = ['time', 'fault_code']
    data_process = DataProcess(origin_data_filename, data_item_list)
    print(data_process.origin_data)
    data_process.to_csv()