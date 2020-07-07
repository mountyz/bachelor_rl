#EXERCICE REPRIS DES TUTORIELS PYTORCH :
# https://pytorch.org/tutorials/beginner/nn_tutorial.html
import pickle
import gzip
import requests
import torch
import torch.nn as nn
import math
from pathlib import Path
from matplotlib import pyplot
import numpy as np

#a revoir je devrais faire en sortes d'être utilisable partout
#ici utilisation de pytorch pour télécharger mnist et avoir le dataset
data_path = Path("M:/users/Document/bachelor_rl/data")
path = data_path / "mnist"

path.mkdir(parents=True, exist_ok=True)

url = "http://deeplearning.net/data/mnist/"
filename = "mnist.pkl.gz"

if not (path / filename).exists():
        content = requests.get(url + filename).content
        (path / filename).open("wb").write(content)

with gzip.open((path / filename).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

#pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")

#_________________________________________________________________________________________________________________

#le dataset récupéré est en numpy il faut alors convert nos datas en tensor
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

#récupère la taille des données
n, c = x_train.shape

x = torch.randn(1,10)
print(x)
print(x.exp())
print(x.exp().sum(-1).log())
print(x.exp().sum(-1).log().unsqueeze(-1))

#initialisation des poids, 784 car l'image est en 28x28 en divisant par la racine carré de 784
weights = torch.randn(784, 10) / math.sqrt(784)

#ici on ajouter le requires_grad sur le tensor des weigths
weights.requires_grad_()

#on créer notre bias (besoin de plus d'inforamtion)
bias = torch.zeros(10, requires_grad=True)

#softmax permet de transformer dans input (0 à 9) en probabilité en prenant l'exponientiel
# et en la divisant par la somme de toutes les autres exponentielles
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

#ici on a notre modèle qui prend le sample en cours et le softmax en utilisant les poids
#le @ fait le dot
def model(x):
    return log_softmax(x @ weights + bias)

#Pas sur du  fonctionnement besoin d'explication
def loss_function_cross_entropy(pred, target):
    return -(target * pred.log()).sum(-1)

def accuracy(out, y):
    preds = torch.argmax(out, dim=1)
    return (preds == y).float().mean()

batch_size = 64
criterion = nn.CrossEntropyLoss()
learning_rate = 0.5
epochs = 10
loss_track = []
accuracy_track = []

for epoch in range(epochs):
    for i in range((n - 1)// batch_size + 1):
        #definition du debut de l'echantillion
        start = i * batch_size

        #definition du fin de l'echantillion
        end = start + batch_size

        #recupération des données en fonction de l'échantillion
        x = x_train[start:end]
        y = y_train[start:end]

        #calcule du y_pred avec le modèle (softmax)
        y_pred = model(x)
        print(y_pred.shape())
        print(y.shape())

        #calcule de la loss avec le y_pred et le y
        #loss = criterion(y_pred, y)
        loss = loss_function_cross_entropy(y_pred, y)
        #print(loss, accuracy(y_pred, y))
        #print(accuracy(y_pred, y))
        #print(loss.detach().numpy())
        if i % 100 == 0:
            loss_track.append(loss.item())
            accuracy_track.append(accuracy(y_pred, y))

        #backpropagation
        loss.backward()

        #update manuel classic avec pytorch
        with torch.no_grad():
            weights -= weights.grad * learning_rate
            bias -= bias.grad * learning_rate
            weights.grad.zero_()
            bias.grad.zero_()

pyplot.plot(loss_track)
pyplot.show()
pyplot.plot(accuracy_track)
pyplot.show()







