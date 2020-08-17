import pandas as pd

epochs = 1000
learning_rate = 0.001
datas = pd.read_csv('data.csv')
X = datas.iloc[:, 0]
Y = datas.iloc[:, 1]

def linear_regression_nothing():
    m = 0
    b = 0
    while 1:
        for i in range(200):
            prediction = m * X + b
            error = prediction - Y
            m = m - learning_rate * (error * X)
            b = b - learning_rate * error
