# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 22:16:51 2023

@author: workf
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def fetch_data(stock, period):
    ticker = stock
    df = yf.download(ticker, period=period)
    df_tail = df.tail(20)
    return df, df_tail

def cleaning_data(df):
    #df.index = df.index.tz_localize(None)
    df = df.interpolate(method='linear', limit_direction='both')
    return df

def labeling(df):
    ilocs_min = argrelextrema(df.Close.values, np.less_equal, order=3)[0]
    ilocs_max = argrelextrema(df.Close.values, np.greater_equal, order=3)[0]
    df.Close.plot(figsize=(20,8), alpha=.3)
    df.iloc[ilocs_max].Close.plot(style='.', lw=10, color='red', marker="v")
    df.iloc[ilocs_min].Close.plot(style='.', lw=10, color='green', marker="^")
    Action = []
    a=df.iloc[ilocs_max].index
    b=df.iloc[ilocs_min].index
    for row in df.index:
      if row in a:
        Action.append(2)
      elif row in b:
        Action.append(1)
      else:
        Action.append(0)
    df['Action'] = Action
    df = df.drop(df[df['Action'] == 0].index)
    return df

def SMA(df):
    df["SMA"] = df["Close"].rolling(window=20).mean()
    return df

def EMA(df):
    df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df

def ADI(df):
    adi = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    adi *= df['Volume']
    adi = adi.shift(1) + adi
    adi = adi.cumsum()
    df['ADI'] = adi
    return df

def MACD(df):
    df["26ema"] = df["Close"].ewm(span=26).mean()
    df["12ema"] = df["Close"].ewm(span=12).mean()
    df["MACD"] = df["12ema"] - df["26ema"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df = df.drop(['26ema', '12ema', 'MACD'], axis=1)
    return df

def Stochastic_Oscillator(df):
    df["L14"] = df["Low"].rolling(window=14).min()
    df["H14"] = df["High"].rolling(window=14).max()
    df["%K"] = 100 * ((df["Close"] - df["L14"]) / (df["H14"] - df["L14"]))
    df["%D"] = df["%K"].rolling(window=3).mean()
    df = df.drop(['L14', 'H14'], axis=1)
    return df

def RSI(df):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def dropper(df):
    df = df.dropna()
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)
    return df

def Split_var(df):
    X = df.drop('Action', axis=1)
    y = df['Action']
    return X,y

def scaler(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def classify(X_train, y_train):
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    return rfc

def evaluate(rfc, X_test, y_test):
    y_pred = rfc.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
    disp.plot()
    # Plot the confusion matrix as a heatmap
    return

def predict(df, rfc):
    df = cleaning_data(df)
    df = SMA(df)
    df = EMA(df)
    #df = ADI(df)
    #df = MACD(df)
    df = Stochastic_Oscillator(df)
    #df = RSI(df)
    df = dropper(df)
    #df = df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)
    X = scaler(df)
    action = rfc.predict(X)
    
    return action

df, df_tail = fetch_data('UNVR.JK', '3y')
df = cleaning_data(df)
df = labeling(df)
df = SMA(df)
df = EMA(df)
#df = ADI(df)
#df = MACD(df)
df = Stochastic_Oscillator(df)
#df = RSI(df)
df = dropper(df)
print(df)
X,y = Split_var(df)
X = scaler(X)
X_train, X_test, y_train, y_test = train_test(X,y)
rfc = classify(X_train, y_train)
evaluate(rfc, X_test, y_test)
aksi = predict(df_tail, rfc)
print(aksi)
    

