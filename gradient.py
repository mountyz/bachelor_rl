#EXERCICE REPRIS DES TUTORIELS PYTORCH : https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as nn

epochs = 1000
learning_rate = 0.001
dtype = torch.float
device = torch.device("cpu")
device = torch.device("cuda:0")


def gradient_descent_pytorch_nn():
    batch_size, input_dim, hidden_layer_size, output_dim = 64, 1000, 100, 10

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)

    model = torch.nn.Sequential(
        nn.Linear(input_dim, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, output_dim),
    )

    loss_fn = nn.MSELoss(reduction='sum')
    #optimizer_fn = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        #optimizer_fn.step()

def gradient_descent_pytorch():
    batch_size, input_dim, hidden_layer_size, output_dim = 64, 1000, 100, 10

    x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    y = torch.randn(batch_size, output_dim, device=device, dtype=dtype)

    w1 = torch.randn(input_dim, hidden_layer_size, device=device, dtype=dtype)
    w2 = torch.randn(hidden_layer_size, output_dim, device=device, dtype=dtype)

    for t in range(epochs):
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        #loss = (y_pred - y).pow(2).sum().item()

        gradient_y_pred = 2.0 * (y_pred - y)
        gradient_w2 = h_relu.T.mm(gradient_y_pred)
        gradient_h_relu = gradient_y_pred.mm(w2.T)
        gradient_h = gradient_h_relu.copy()
        gradient_h[h < 0] = 0
        gradient_w1 = x.T.mm(gradient_h)

        w1 -= learning_rate * gradient_w1
        w2 -= learning_rate * gradient_w2

def gradient_descent_numpy():
    batch_size, input_dim, hidden_layer_size, output_dim = 64, 1000, 100, 10

    x = np.random.randn(batch_size, input_dim)
    y = np.random.randn(batch_size, output_dim)

    w1 = np.random.randn(input_dim, hidden_layer_size)
    w2 = np.random.randn(hidden_layer_size, output_dim)

    for i in range(epochs):
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        #loss = np.square(y_pred - y).sum()

        gradient_y_pred = 2.0 * (y_pred - y)
        gradient_w2 = h_relu.T.dot(gradient_y_pred)
        gradient_h_relu = gradient_y_pred.dot(w2.T)
        gradient_h = gradient_h_relu.copy()
        gradient_h[h < 0] = 0
        gradient_w1 = x.T.dot(gradient_h)

        w1 -= learning_rate * gradient_w1
        w2 -= learning_rate * gradient_w2

def linear_regression_nothing():
    datas = pd.read_csv('data.csv')
    X = datas.iloc[:, 0]
    Y = datas.iloc[:, 1]
    m = 0
    b = 0
    i = 0

    while 1:
        for i in range(epochs):
            prediction = m * X + b
            error = prediction - Y
            m = m - learning_rate * (error * X)
            b = b - learning_rate * error

        plt.scatter(X, Y)
        plt.plot([min(X), max(X)], [min(prediction), max(prediction)], color='red')
        plt.show()

        i += 1
        if i % 5 == 0:
            print(i*1000, "epochs elapsed")

            stop = input("stop ? y")
            if stop == "y":
                break