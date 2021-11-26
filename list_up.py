#파일 목록 csv로 저장

import os
import pandas as pd
import numpy as np

path_dir = './Train_1h/'

folder_list = os.listdir(path_dir)

DF = pd.DataFrame()

for folder in folder_list:
    if ".ini" in folder: continue
    print(folder)
    file_list = np.array(os.listdir(path_dir+folder))
    file_list = file_list[".csv" in file_list]
    DF[folder] = file_list

DF.to_csv("1h_list.csv", mode='w', engine="python")
