import time

import requests
import pandas as pd
key = "4345bb3fa4d57793f9a569c5ce8f72cc"
source_data_file = "../211223Res/gps.txt"
def api_test(location_info):
    api = "https://restapi.amap.com/v3/geocode/regeo?key={0}&location={1}&batch=true".format(key, location_info)
    response = requests.get(api)
    result = response.json()
    print(response.status_code)
    for item in result["regeocodes"]:
        print(item['formatted_address'])
        info = item["addressComponent"]
        print(info["province"])
        print(info["city"])

def remove_zero():
    data = pd.read_csv(source_data_file)
    print(data)
    e_list = data['E']
    n_list = data['N']
    drop_list = []
    for i in range(len(e_list)):
        if e_list[i] == 0 or n_list[i] == 0:
            drop_list.append(i)
    data = data.drop(index=drop_list)
    data.to_csv("gps.txt", index=False)

if __name__ == "__main__":
    # remove_zero()
    data = pd.read_csv("gps.txt")
    l = len(data)
    times = 0
    Adress_list = []
    Province_list = []
    City_list = []
    for i in range(0, l, 20):
        target_E = data['E'][i: min(i + 20, l)]
        target_N = data['N'][i: min(i + 20, l)]
        location_str = str(format(target_E[i], '.6f')) + "," +str(format(target_N[i], '.6f'))
        for j in range(i + 1, i + len(target_E)):
            location_str = location_str + "|"
            location_str = location_str + str(format(target_E[j], '.6f')) + "," + str(format(target_N[j], '.6f'))
        api = "https://restapi.amap.com/v3/geocode/regeo?key={0}&location={1}&batch=true".format(key, location_str)
        response = requests.get(api)
        print(api)
        result = response.json()
        if result['status'] == 0:
            print("出现失败请求!")
            exit(1)
        print(times)
        for item in result["regeocodes"]:
            Adress_list.append(item["formatted_address"])
            info = item["addressComponent"]
            Province_list.append(info["province"])
            City_list.append(info["city"])
        times = times + 1
        if times > 195:
            time.sleep(1.2)
            times = 0
    data["Adress"] = Adress_list
    data["Province"] = Province_list
    data["City"] = City_list
    data.to_csv("target.txt", index=False)



    # test_info = "113.99315,22.64075|113.27665,22.97215"
    # api_test(test_info)