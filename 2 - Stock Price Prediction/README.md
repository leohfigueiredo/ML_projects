# Stock Price Prediction Project

## Introduction
This project focuses on predicting stock prices using historical data of Tesla Inc. The dataset is obtained from Yahoo Finance and includes information on opening, high, low, closing prices, adjusted closing prices, and trading volumes.

## Dataset
To get started, download the historical stock price dataset from Yahoo Finance using the following link:
```python
url = 'https://query1.finance.yahoo.com/v7/finance/download/TSLA?period1=1644247326&period2=1675783326&interval=1d&events=history&includeAdjustedClose=true'
dataset_train = pd.read_csv(url)
```

### Dataset Summary
```python
dataset_train.head()
```
```
         Date        Open        High         Low       Close   Adj Close    Volume
0  2022-02-07  307.929993  315.923340  300.903320  302.446655  302.446655  60994500
1  2022-02-08  301.843323  308.763336  298.266663  307.333344  307.333344  50729100
2  2022-02-09  311.666656  315.423340  306.666656  310.666656  310.666656  52259400
3  2022-02-10  302.790009  314.603333  298.899994  301.516663  301.516663  66126900
4  2022-02-11  303.209991  305.320007  283.566681  286.666656  286.666656  79645800
```

```python
dataset_train.shape
```
```
(252, 7)
```

```python
dataset_train.describe()
```
```
             Open        High         Low       Close   Adj Close        Volume
count  252.000000  252.000000  252.000000  252.000000  252.000000  2.520000e+02
mean   245.512976  251.597315  238.697943  244.913670  244.913670  9.731395e+07
std     62.943779   63.802307   61.806732   62.664325   62.664325  4.485044e+07
min    103.000000  111.750000  101.809998  108.099998  108.099998  4.186470e+07
25%    206.270000  212.349998  198.107498  203.012504  203.012504  6.755422e+07
50%    248.646667  254.903328  240.593331  245.618332  245.618332  8.465205e+07
75%    295.083336  300.209999  285.850007  292.120003  292.120003  1.050083e+08
max    378.766663  384.290009  362.433319  381.816681  381.816681  3.056321e+08
```

```python
dataset_train.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 252 entries, 0 to 251
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Date       252 non-null    object 
 1   Open       252 non-null    float64
 2   High       252 non-null    float64
 3   Low        252 non-null    float64
 4   Close      252 non-null    float64
 5   Adj Close  252 non-null    float64
 6   Volume     252 non-null    int64  
dtypes: float64(5), int64(1), object(1)
memory usage: 13.9+ KB
```

## Exploratory Data Analysis (EDA)
Let's visualize the closing prices to gain insights into Tesla's stock price trends.

```python
import matplotlib.pyplot as plt
import seaborn as sb

# EDA Plot
plt.figure(figsize=(15, 5))
plt.plot(dataset_train['Close'])
plt.title('Tesla Close Price', fontsize=15)
plt.ylabel('Price in Dollars')
plt.show()
```

This plot provides a visual representation of Tesla's closing prices over the given time period, helping to identify patterns and trends in the stock's behavior.

Feel free to explore further and apply various machine learning models for predicting future stock prices based on historical data.
