import numpy as np
import pandas as pd
import time

#数据统计处理
origin_data_filename = "../data/fault.txt"
csv_file_name = '../data/test.csv'
class DataProcess():
    TIMEFORMAT = "%Y-%m-%d %H:%M:%S"
    DATAITEM = ["time", "car_code", "fault_code", "is_master", "fault_pattern", "fault_state"]
    DATAITEMINDEX = {"is_master": 0, "fault_code": 1, "car_code": 2, "time": 3, "fault_pattern": 4, "fault_state": 5}

    def __init__(self,filename, data_item = DATAITEM, generate_csv = True, csv_filename = csv_file_name):
        if generate_csv:
            self.origin_data = self.readtxt(filename, data_item)
            self.to_csv(csv_filename)
        else:
            self.origin_data = pd.read_csv(csv_filename)
            print(self.origin_data)
    def convert_time(self, timestamp):
        loca_time = time.localtime(int(timestamp))
        return time.strftime(DataProcess.TIMEFORMAT, loca_time)

    def convert_timestamp(self, timestr):
        timeArray = time.strptime(timestr, "%Y-%m-%d %H:%M:%S")
        return int(time.mktime(timeArray))

    def get_carriage(self, car_code):
        return str(car_code)[-2:]

    def get_car(self, car_code):
        return str(car_code)[:-2]

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

    def divide_by_car_code(self):
        self.origin_data.insert(self.origin_data.shape[1], 'carriage', self.origin_data['car_code'].apply(lambda x: self.get_carriage(x)))
        self.origin_data['car_code'] = self.origin_data['car_code'].apply(lambda x: self.get_car(x))
        self.car_group_data = self.origin_data.groupby('car_code')

    def count_nums_by_time(self, time_interval = 30):
        time_interval = time_interval * 60
        self.origin_data['time'] = self.origin_data['time'].apply(lambda x: self.convert_timestamp(x))
        time_serials = self.origin_data['time'].values.tolist()
        serials_len = len(time_serials)
        self.counts_serial = []
        i = 0
        while i < serials_len:
            if time_serials[-1] - time_serials[i] < time_interval:
                break
            cou = 0
            j = i
            while(time_serials[j] - time_serials[i] <= time_interval):
                cou = cou + 1
                j = j + 1
            i = j
            self.counts_serial.append(cou)
        print(len(self.counts_serial))
        print(self.counts_serial)

if __name__ == "__main__":
    data_item_list = ['time', 'fault_code', 'car_code']
    data_process = DataProcess(origin_data_filename, data_item_list, generate_csv=False)
    data_process.count_nums_by_time()