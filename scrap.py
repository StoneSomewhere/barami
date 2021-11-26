import pandas as pd
import yfinance as yf
from tqdm import tqdm

List = pd.read_csv('./3bull_list.csv')["list"]
ss = " ".join(List)
Ticker = " ".join(ss.split())

#모두 동시에 다운로드
Data = yf.download(tickers=Ticker, interval="1h", period="max")


# Ticker = Ticker.split()

List = Data["Adj Close"].columns.values
#print(List)


Col = ["High", "Low", "Adj Close", "Volume"]

for name in tqdm(List):
    stock = pd.concat([Data["High"][name], Data["Low"][name], Data["Adj Close"][name], Data["Volume"][name]], axis=1)
    stock.columns = Col
    if stock["Adj Close"].isnull().all():
        i = 700
        while (stock["Adj Close"].isnull().all()):
            stock = yf.download(tickers=name, interval='1h', period=str(i) + 'd')[Col]
            i = i - 50
            if i<300: break

        if stock["Adj Close"].isnull().all(): continue
    stock = stock.dropna(axis=0)
    stock.dropna(subset=['Adj Close'], inplace=True)
    start = str(stock.index[0])[:10]
#    stock.to_csv("./3bull_all/" + name + "_" + "1h" + "_" + start + ".csv", mode='w')
    stock.to_csv("./3bull_1h/" + name + "_" + "1d" + "_" + ".csv", mode='w')  #날짜 안 씀