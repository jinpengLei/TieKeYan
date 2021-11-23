import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pylab import *
#数据统计处理
origin_data_filename = "../data/fault.txt"
csv_file_name = '../data/cars/1004.csv'
class DataProcess(object):
    TIMEFORMAT = "%Y-%m-%d %H:%M:%S"
    DATAITEM = ["time", "car_code", "fault_code", "is_master", "fault_pattern", "fault_state"]
    DATAITEMINDEX = {"is_master": 0, "fault_code": 1, "car_code": 2, "time": 3, "fault_pattern": 4, "fault_state": 5}

    def __init__(self, filename, data_item=DATAITEM, generate_csv=True, csv_filename=csv_file_name):
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
        print(dataframe)
        return dataframe

    def to_csv(self, target_file_name='../data/test.csv'):
        self.origin_data.to_csv(target_file_name, index=False, sep=',')

    def divide_by_car_code(self):
        self.origin_data.insert(self.origin_data.shape[1], 'carriage', self.origin_data['car_code'].apply(lambda x: self.get_carriage(x)))
        self.origin_data['car_code'] = self.origin_data['car_code'].apply(lambda x: self.get_car(x))
        self.car_group_data = self.origin_data.groupby('car_code')
        cou = 0
        Max_nums = 0
        t = ""
        nums_list = []
        car_code_list = []
        for car_code, group in self.car_group_data:
            out_put_filename = '../data/cars/' + car_code + '.csv'
            print(car_code)
            print(group)
            nums_list.append(len(group))
            car_code_list.append(car_code)
            if(len(group) > Max_nums):
                Max_nums, t = len(group), car_code
            group.to_csv(out_put_filename, index=False, sep=',')
        df = pd.DataFrame({'car_code': car_code_list, 'nums': nums_list})
        df = df.sort_values(by='nums', ascending=False)
        df = df[:10]
        print(df)
        print(Max_nums, t)



    def count_nums_by_time(self, time_interval=30):
        time_interval = time_interval * 60
        self.origin_data['time'] = self.origin_data['time'].apply(lambda x: self.convert_timestamp(x))
        time_serials = self.origin_data['time'].values.tolist()
        serials_len = len(time_serials)
        self.counts_serial = []
        self.time_serial = []
        i = 0
        start_time = time_serials[0]
        end_time = time_serials[0] + time_interval
        while i < serials_len:
            if time_serials[-1] < end_time:
                break
            cou = 0
            j = i
            while(time_serials[j] - start_time <= time_interval):
                cou = cou + 1
                j = j + 1
            self.counts_serial.append(cou)
            self.time_serial.append(start_time)
            start_time = end_time
            end_time = end_time + time_interval
            i = j
        print(len(self.counts_serial))
        print(self.counts_serial)
        target_csv_filename = '../data/counts_serial.csv'
        df = pd.DataFrame({'time': self.time_serial, 'fault_nums': self.counts_serial})
        df['time'] = df['time'].apply(lambda x: self.convert_time(x))
        df.to_csv(target_csv_filename, index=False, sep=',')
        plt.plot(df['time'], df['fault_nums'])
        plt.show()

    def count_nums_by_day(self):
        self.origin_data['day'] = self.origin_data['time'].apply(lambda x: "%s" % x[:10])
        self.nums_by_days = self.origin_data.groupby('day').size()
        self.nums_by_days.to_csv("../data/1004_nums_by_day.csv")
        # df['Week/Year'] = df['Timestamp'].apply(lambda x: "%d/%d" % (x.week, x.year))

class BigDataProcess(object):
    fault_path = "E:/TieKeYan/data/fault"
    parameter_path = "E:/TieKeYan/data/parameter"
    gps_path = "E:/TieKeYan/data/gps"
    self_check_path = "E:/TieKeYan/data/selfcheck"
    def __init__(self, txtfilename):
        self.txtfilename = txtfilename

    def split_by_car_code(self):
        with open(self.txtfilename, 'r') as f:
            line = f.readline()
            cou = 1
            target_code = '1004'
            while line:
                per_line_list = line[4:].split('|')
                car_code = per_line_list[5]
                flag = True
                if line[:2] == '03' and car_code == target_code:
                    filename = self.parameter_path + "/" + target_code
                elif (line[:2] == '01' or line[:2] == '02') and car_code == target_code:
                    filename = self.fault_path + "/" + target_code
                elif line[:2] == '04' and car_code == target_code:
                    filename = self.self_check_path + "/" + target_code
                elif line[:2] == '05' and car_code == target_code:
                    filename = self.gps_path + "/" + target_code
                else:
                    flag = False
                if flag:
                    filename = filename + '.txt'
                    with open(filename, 'a+') as f1:
                        f1.write(line)
                cou = cou + 1
                line = f.readline()
            print(cou)


if __name__ == "__main__":
    # bigdataprocess = BigDataProcess("E:/TieKeYan/113-66.txt")
    # bigdataprocess.split_by_car_code()
    data_item_list = ['time', 'fault_code', 'car_code']
    data_process = DataProcess(origin_data_filename, data_item_list, generate_csv=False)
    data_process.count_nums_by_time(time_interval=30)