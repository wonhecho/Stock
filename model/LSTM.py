import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import FinanceDataReader as fdr
# 삼성전자의 2009년 9월 16일 정보부터 2021-07-28일 까지의 주가 정보를 가져온다.
price_df = fdr.DataReader("005930",'2009-09-16','2021-07-28')

# Date 부분이 index로 되어있기 때문에 이를 칼럼으로 옮겨준다.
price_df = price_df.rename_axis('Date').reset_index()
# Date 부분의 포멧을 Datetime으로 변경해준다.
pd.to_datetime(price_df['Date'],format='%Y.%m.%d')
#식별하기 편하도록 column의 이름들을 한글로 바꿔줌.
price_df = price_df.rename(columns = {"Date":"일자","Open":"시가","High":"고가","Low":"저가","Close":"종가","Volume":"거래량"})
price_df['일자'] = pd.to_datetime(price_df['일자'], format='%Y.%m.%d')
price_df['연도'] =price_df['일자'].dt.year
price_df['월'] =price_df['일자'].dt.month
price_df['일'] =price_df['일자'].dt.day
#price_df.rename(columns = {"Date":"일자","Open":"시가","High":"고가","Low":"저가","Close":"종가","Volume":"거래량"})

#이를 차트화하여 확인한다. 종가기준 y축 일자 x축
df = price_df.loc[price_df['연도']>=2009]
plt.figure(figsize=(16, 9))
sns.lineplot(y=df['종가'], x=df['일자'])
plt.xlabel('time')
plt.ylabel('price')
# sklearn라이브러리의 MinMaxScaler 스케일링을 이용 StandardScaler, MaxAbsScaler등 사용 가능
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 스케일링에 들어갈 칼럼들을 설정 및 변환과정
scale_cols = ['시가', '고가', '저가', '종가', '거래량']
df_scaled = scaler.fit_transform(df[scale_cols])
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

window_size = 20
TEST_SIZE = 200
train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
  
import numpy as np
feature_cols = ['시가', '고가', '저가', '거래량']
label_cols = ['종가']

train_feature = train[feature_cols]
train_label = train[label_cols]
# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

#x_train.shape, x_valid.shape

# test dataset (실제 예측 해볼 데이터)
test_feature = test[feature_cols]
test_label = test[label_cols]
test_feature, test_label = make_dataset(test_feature, test_label, 20)
#test_feature.shape, test_label.shape

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# LSTM 모델 생성
model = models.Sequential()
model.add(layers.LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)
filename = 'tmp_checkpoint.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train, 
                    epochs=50, 
                    batch_size=16,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])

model.load_weights(filename)

# 예측
pred = model.predict(test_feature)
# 결과 그래프 
plt.figure(figsize=(16, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
sns.lineplot(y=df['종가'])
plt.ylabel('price')
plt.legend()
plt.show()
