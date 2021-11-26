---
title: 거래량 & 매매가로 등락 예측
author: YoungchanJo
date: 2021-11-26
categories: [Exhibition,2021년]
tags: [post,YoungchanJo, Quant] 
---

# 거래량 & 매매가로 개별 주식 등락 예측

### 작품소개(선행 연구와 연구 방향) 

딥러닝을 이용한 주가 예측은 크게 3가지 흐름이 있다.
1. 거시지표(물가, 유가, 환율, GDP, 시장지수 등)를 이용하여 시장전체 지수의 변동을 예측
2. 개별 주식 종목의 재무정보(PER, PBR, PCR 등)와 거래정보(거래량, 종가 등)를 이용하여 주가를 예측
3. 개별 주식 종목의 거래정보(거래량, 거래가)를 이용하여 주가를 예측

  1에 대한 선행 연구들을 살펴본 결과 지수의 변동은 장기적으로 볼 때 70%를 넘는 정확도의 예측도 가능한 것으로 보인다. 그러나 이런 정보를 이용하여 투자하기는 어렵다. 지수 ETF에 투자할 것이 아닌 이상 지수와 개별종목은 같은 방향으로 움직이지 않으며, 지수 ETF의 경우 지나치게 긴 시간을 투자해야하기 때문이다. 긴 시간의 문제는 우리가 인내하는 동안 시장의 변동 원리가 변하여 우리의 모델 자체가 무효화되었을 가능성을 배제할 수 없다는 점이다.

  2에 대한 선행 연구들을 살펴본 결과 재무정보를 이용한 예측은 생각보다 정밀도가 떨어진다. 거기다가 재무정보들은 분기별로 제공되므로 연속적인 예측에 신뢰성을 담보하지 않는다. 대체로 재무정보는 장기적인 추세에서는 유의미하나 단기적인(분기내에서의) 변동을 예측하는데는 쓸모가 없었다.

  3에 대한 선행 작업들의 특징이 있다. a) 예측된 주가가 무엇을 의미하는지 알 수가 없다. 단순하게 생각해서 최소제곱오차로 구현했다고 가정하자. 오차의 최소는 등락을 정확하게 맞춤을 의미하지 않는다. 부분적으로 틀리더라도 전체적인 추세를 맞추면 loss는 내려간다. 그런데 문제는 loss의 의미를 확률로 치환하여 투자에 사용할 수 없다는 점이다. b) 단일 종목 또는 3~4 종목에 대한 성과가 대부분이다. 본인은 단일 종목으로 80%넘는 정확도를 달성한 것들도 있다. 종목의 수가 적은 결과는 믿을 수가 없다. 다시 말해서 일반화가 되지 않은 결과를 믿고 투자할 수가 없다는 의미이다. 올바른 모델이라면 주식 전체에 대해서도 합리적인 수준의 예측을 할 수 있어야 한다.

  위와 같은 이유로 나는 거시지표와 재무정보를 배제하였고, 주가를 예측하는 것이 아니라 주식이 n%이상 오를지 또는 내릴지를 기준으로 삼았으며, 2천개의 주식과 1천개의 etf를 학습데이터로 사용하였다.


### 작품소개(개략적인 작동방식)

  매우 실험적인 시도를 많이 진행하였으므로 구체적인 프로그램을 만들었다기 보다는 코드를 수정해가며 필요한 것을 파악해나갔다. 때때로 엑셀을 쓰기도 했고, 윈도우 파워쉘을 사용하기도 했다. 본인은 이전 버전 저장을 하지 않는 아주 나쁜 습관을 가지고 있어, 매우 많은 자료가 유실되었다. 

  전체적으로 이런 방식으로 연구했다. 종목코드를 *_list.csv로 저장 > yfinance와 판다스 라이브러리를 이용하여 각 종목별로 종목이름.csv로 저장 > 가공하기 쉽게 처리한 후 pickle로 저장 > 학습시킬 데이터 구조에 맞게 X_***.npy, Y_***.npy로 저장 > RNN계열 모델로 학습 > 적당히 즉석해서 코드를 짜서 테스트

  데이터의 기본적으로 1시간 단위로 잡았고, n%이상 오르면 1 내리면 0으로 라벨링했다. 주식, ETF, 비트코인을 학습 데이터로 이용하였고, 3배 레버리지 ETF를 검증 데이터로 사용하였다. 3배 레버리지 ETF를 사용한 이유는 (일확천금을 노려서) 특별히 없다.



### 데이터 가져오기
아래는 데이터를 가져오는 코드이다. 특별한 설명보다는 상세히 주석을 달았다.

import pandas as pd
import yfinance as yf
from tqdm import tqdm

List = pd.read_csv('./3bull_list.csv')["list"]
ss = " ".join(List)
Ticker = " ".join(ss.split())	#*.csv에 있는 종목코드가 공백으로 구분되는 문자열이 된다.

#Ticker가 종목코드, interval은 1h, 1d, 1y등이 가능하다. 병렬 다운방식이다.
Data = yf.download(tickers=Ticker, interval="1h", period="max")

'''
아래 내용은 이중 컬럼 때문에 직관적으로 이해하기 어려울 수 있다. 그냥 한꺼번에 다운 받은 후 나눠서 csv로 저장한다고 생각하면 된다. 
'''
List = Data["Adj Close"].columns.values


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
#다운로드가 실패할 경우 기간을 i로 잡아서 다시 다운로드를 시도한다.

        if stock["Adj Close"].isnull().all(): continue
    stock = stock.dropna(axis=0)
    stock.dropna(subset=['Adj Close'], inplace=True)
    start = str(stock.index[0])[:10]
    stock.to_csv("./3bull_all/" + name + "_" + "1h" + "_" + start + ".csv", mode='w')


### 데이터 가져오기(피클로 전환)
대용량 데이터를 처리함에 있어서 중복처리로 인한 시간 낭비를 줄이기 위해 피클을 사용했다. folder_list는 적당히 바꿔서 썼다. 원래 인덱스는 YYYY-MM-DD HH:30-HH:30이라는 매우 복잡한 형태인데, 이것을 YYYYMMDDHH로 정수화했다.

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

path_dir = './'
#folder_list = ['Bitcoin_1h', 'stock_1h', 'etf_1h', 'stock_1d']
folder_list = ['3bull_1d', '.ini']

for folder in folder_list:
    if ".ini" in folder: continue
    print(folder)
    X={}
    
    file_list = os.listdir(path_dir + folder)
    
    for file in tqdm(file_list):
        if ".csv" not in file: continue
        
        name=file.split('_')[0]
        
        file = path_dir + folder + "/" + file

        Data = pd.read_csv(file, index_col=0, engine='python')
        Data.dropna(subset=['High'], inplace=True)
        Data.index = [int(''.join(x.split(':')[:2]).replace('-', '').replace(' ', '')) for x in Data.index]  #인덱스 정수화
        
        X[name]=Data
        
    with open(folder+".pickle","wb") as fw:
        pickle.dump(X, fw)



### 결과설명

  비트코인은 주식보다 예측 정확도가 낮았다. 대신 정밀도(loss)가 더 좋았다. 비트코인의 최대 정확도는 55.9%였고 loss는 focal loss(gamma=1)기준 0.3294였다. 이 때의 데이터 구조는 (64,4)였고, 사용한 데이터는 최고가-최저가, log(거래량), log(거래량)*종가, 종가였다.
  주식(3배 ETF은 정확도가 높은 대신 정밀도가 나빴다. 최대 정확도는 62.98%였고 이 때의 정밀도는 0.3346이었다. 데이터 구조는 (280, 4)였고, 사용한 데이터는 최고가-최저가, log(거래량)*종가, 종가, 거래량이었다. loss를 줄이기 위해 다양한 시도를 했지만 본인이 했던 실험에서는 위 구조가 비트코인과 주식에 대한 최적의 구조였다.





### 결과 및 느낀점 

  개인적으로 굉장히 많은 노력을 투자했지만 loss가 나빠서 투자에 사용할 수 없다고 결론내렸다. 딥러닝을 이용한 단기 주가 예측은 쓸모가 없는 것 같다. 아무리 정확도를 높여봤자 loss가 높으면 예측 결과의 신뢰성이 떨어져서 투자에 사용할 수 없다. 50.1%의 확률로 오른다고 예측하고 99%의 확률로 떨어진다고 예측했는데 둘다 오른다면 예측을 믿을 수 있겠는가?
  따라서 머신러닝을 이용한 단기 주식 예측보다는 장기 주식 예측과 포트폴리오 이론을 이용하기 위한 베타분석을 하는 것이 더 타당하다고 본다.

  결과가 기대 이하라서 자세하게 설명할 인내심이 생기지 않는다. 사용한 모든 자료는 github에 올렸으니 궁금한 점을 이 게시글에 문의하면 2022/03까지는 성실히 답변하겠다.

https://github.com/StoneSomewhere/barami


### 참고자료

- https://github.com/artemmavrin/focal-loss
- https://www.tensorflow.org/guide/keras/transfer_learning?hl=ko
- https://conanmoon.medium.com/%EB%8D%B0%EC%9D%B4%ED%84%B0%EA%B3%BC%ED%95%99-%EC%9C%A0%EB%A7%9D%EC%A3%BC%EC%9D%98-%EB%A7%A4%EC%9D%BC-%EA%B8%80%EC%93%B0%EA%B8%B0-%EC%97%B4%EC%97%AC%EC%84%AF-%EB%B2%88%EC%A7%B8-%EC%9D%BC%EC%9A%94%EC%9D%BC-8a6cc162fd8
- https://parkeunsang.github.io/blog/stock/2021/06/11/python-api.html
- https://www.nanumtrading.com/fx-%eb%b0%b0%ec%9a%b0%ea%b8%b0/%ec%b0%a8%ed%8a%b8-%eb%b3%b4%ec%a1%b0%ec%a7%80%ed%91%9c-%ec%9d%b4%ed%95%b4/01-%eb%b3%b4%ec%a1%b0%ec%a7%80%ed%91%9c%eb%9e%80/
- https://jangminhyeok.tistory.com/4?category=726428
- https://jangminhyeok.tistory.com/4?category=726428
- https://elzino.github.io/papers/2019-11-21/radam
- 
