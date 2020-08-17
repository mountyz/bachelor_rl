#EXERCICE REPRIS DES TUTORIELS PYTORCH :
# https://pytorch.org/tutorials/beginner/nn_tutorial.html
import pickle
import gzip
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import math
from pathlib import Path
from matplotlib import pyplot
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

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

def accuracy_fct(out, y):
    preds = torch.argmax(out, dim=1)
    return (preds == y).float().mean()

batch_size = 64

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)

#pour bénéficier des foncionnalité d'optimisation,modèle utilisant nn.Module requis
class mnist_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, x):
        return self.lin(x)

def get_model():
    model = mnist_model()
    return model, optim.SGD(model.parameters(), lr=learning_rate)

valid_accuracy_track = []
accuracy_track = []
loss_track = []
valid_loss_track = []
learning_rate = 0.5
epochs = 10
criterion = nn.CrossEntropyLoss()
model, opt = get_model()
i = 0

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = criterion(pred, yb)
        accuracy = accuracy_fct(pred, yb)
        if i % 64 == 0:
            loss_track.append(loss)
            accuracy_track.append(accuracy)
        i += 1
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        for x, y in valid_dl:
            valid_pred = model(x)
            valid_loss = criterion(valid_pred, y)
            valid_accuracy = accuracy_fct(valid_pred, y)
            if i % 10 == 0:
                valid_loss_track.append(valid_loss)
                valid_accuracy_track.append(valid_accuracy)
            i += 1
        valid_loss_overall = sum(criterion(model(xb), yb) for xb, yb in valid_dl)

    print("round over the dataset :",epoch, "the means loss is", valid_loss_overall / len(valid_dl))

pyplot.plot(valid_loss_track, label="valid_loss")
pyplot.plot(loss_track, label="loss")
pyplot.legend(loc=1)
pyplot.show()
pyplot.plot(valid_accuracy_track, label="valid_accuracy")
pyplot.plot(accuracy_track, label="accuracy")
pyplot.legend(loc=1)
pyplot.show()






