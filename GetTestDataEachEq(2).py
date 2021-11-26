import os
import pandas as pd
import numpy as np
from tqdm import tqdm


save_path = "./test/"
path_dir = './'
# folder_list = os.listdir(path_dir)
folder_list = ("3bull_1h", ".ini")


base = 128      #몇 일의 데이터를 쓸 것인가
period = 21     #몇 step내를 예측할 것인가
step = 1        #몇 step내를 예측할 것인가

for folder in folder_list:
    if ".ini" in folder: continue
    print(folder)
    file_list = np.array(os.listdir(path_dir + folder))
    file_list = np.flip(file_list)

    for file in tqdm(file_list):
        if ".csv" not in file: continue
        name = file.split("_")[0]
        file = path_dir  + folder + "/" + file
        X, Y = [], []

        Data = pd.read_csv(file, index_col=0, engine='python')
        Data.index = [int(x.split(':')[0].replace('-', '').replace(' ', '')) for x in Data.index]   #인덱스를 정수로 바꿈
        Data.columns = ["High", "Low", "Close", "Volume"]
        Data["Volume"]=np.log(Data["Volume"] + 1)
        Data.dropna(subset=['Close'], inplace=True)
        #        print(Data)

        #date_list = Data.loc['2021090415':].index[:-period]   #예측하려는 날짜-1
        date_list = Data.loc['2021083109':].index[:-period]   #예측하려는 날짜-1
        #date_list = Data.iloc[base-1:].index[:-period]     #예측하려는 날짜-1


        #i는 예측하려는 날-1
        for i in date_list:

            close_std = Data.loc[:i].iloc[-base:]["Close"].std()  # 수정종가의 표준편차
            Last_price = Data.loc[i]['Close'] + 0.0001
            y = Data.loc[i:].iloc[1:period + 1]["Close"] / Last_price - 1
            err = 1

            for s in range(8, 6, -1):  # 7%는 올라야지
                fluct = y[abs(y) > 0.01 * s]
                if np.shape(fluct)[0] > 0:
                    y = fluct.iloc[0]  # 후에 어떤 추가적 변동이 있었든지 가장 먼저 변동한 것만 취급한다.
                    err = 0
                    break
            if err == 1: continue

            x = Data.loc[:i].iloc[-base:]  # 예측일자 -base일
            if x.isnull().values.any(): continue  # null값 있으면 pass
            x = (x - x.mean()) / (x.std() + 0.0001)  # 정규화

            y = (np.sign(y) + 1) // 2  # 올랐으면 1 내렸으면 0
            Y.append(y)
            X.append(x)



            # NaN 제거
        nan_list = np.isnan(X)
        nan_index = [i for i in range(len(nan_list)) if bool(nan_list[i].any())]
        X = np.delete(X, nan_index, axis=0)
        Y = np.delete(Y, nan_index)

        X = np.array(X)
        Y = np.array(Y)

        np.save(save_path+'X_%s.npy'%(name), X)
        np.save(save_path+'Y_%s.npy'%(name), Y)

