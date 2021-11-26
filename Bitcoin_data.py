import requests
from datetime import datetime
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


result = requests.get('https://api.binance.com/api/v3/ticker/price')
js = result.json()
symbols = [x['symbol'] for x in js]
symbols_usdt = [x for x in symbols if 'USDT' in x]  # 끝이 USDT로 끝나는 심볼들, ['BTCUSDT', 'ETHUSDT', ...]


def get_data(start_date, end_date, symbol):
    COLUMNS = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'quote_av', 'trades',
               'tb_base_av', 'tb_quote_av', 'ignore']

    data = []

    start = int(time.mktime(datetime.strptime(start_date + ' 00:00', '%Y-%m-%d %H:%M').timetuple())) * 1000
    end = int(time.mktime(datetime.strptime(end_date + ' 23:59', '%Y-%m-%d %H:%M').timetuple())) * 1000
    params = {
        'symbol': symbol,
        'interval': '5m',
        'limit': 1000,
        'startTime': start,
        'endTime': end
    }

    while start < end:
        params['startTime'] = start
        result = requests.get('https://api.binance.com/api/v3/klines', params=params)
        js = result.json()
        if not js:
            break
        data.extend(js)  # result에 저장
        start = js[-1][0] + 60000  # 다음 step으로
    # 전처리
    if not data:  # 해당 기간에 데이터가 없는 경우
        print('해당 기간에 일치하는 데이터가 없습니다.')
        return -1
    df = pd.DataFrame(data)
    df.columns = COLUMNS
    df['Open_time'] = df.apply(lambda x: datetime.fromtimestamp(x['Open_time'] // 1000), axis=1)
    df.index = df['Open_time']
    df = df[['High', 'Low', 'Close', 'Volume']]

    return df


start_date = '2000-09-10'
end_date = '2021-08-29'



for symbol in tqdm(symbols_usdt[::-2]):

    Data = get_data(start_date, end_date, symbol)
    if(np.shape(Data)):
        Data.dropna(axis=0).to_csv("./Bitcoin_5m/"+symbol+"_"+"5m"+".csv", mode='w')

print("Done!")