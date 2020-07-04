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


def gradient_descent_pytorch_using_nn():
    batch_size, input_dim, hidden_layer_size, output_dim = 64, 1000, 100, 10

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)

    #ici on définit notre réseaud de neuronne. on voit ici une première couche linaire avec les input
    #un relu et une dernière couche lineair avec les outputs
    model = torch.nn.Sequential(
        nn.Linear(input_dim, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, output_dim),
    )

    #on définit la loss on moyen de la fonction nn qui récupère directement la fonction et son traitement
    loss_fn = nn.MSELoss(reduction='sum')
    #on définit l'optimizer qui permet de rentre l'update des poids automatique dans un facon spécifique ici Adam mais possible SGD ou autre
    #optimizer_fn = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        #ici nos input rentre dans le model
        y_pred = model(x)

        #on définit la fonction de loss avec le y_pred ainsi que les outputs
        loss = loss_fn(y_pred, y)

        #on met les gradients à 0 avant de faire le backward pass
        model.zero_grad()

        #calcule le gradient de la loss en fonction de tout les parametres possible du model.
        # à l'interne les paramètres de tous les modules ont le requires_grad=True
        # donc cette appel calculera le gradients de tout les paramètres du model
        loss.backward()

        #soit on fait manuellement comme précedement
        # mais la vu que notre model nous retour tout les paramètres on les récupères directement
        # grace à parametre() et on update les poids en fonctzion du learning rate
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        #ou alors grace à l'optimizer on effectue notre updates
        #optimizer_fn.step()

def gradient_descent_pytorch_control_over_weigths():
    batch_size, input_dim, hidden_layer_size, output_dim = 64, 1000, 100, 10

    x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    y = torch.randn(batch_size, output_dim, device=device, dtype=dtype)

    w1 = torch.randn(input_dim, hidden_layer_size, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(hidden_layer_size, output_dim, device=device, dtype=dtype, requires_grad=True)

    for i in range(epochs):
        #comme avant définition de y_pred et étant donnée qu'on utilise la fonction backwards pas besoin de stocker les variable comme h ou h_relu
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        #défition de la fonction de loss: MSELoss : la somme de la la prediction de y - le vrai y a la puissance 2
        loss = (y_pred - y).pow(2).sum()

        #ici on utilise l'autograd grace à la fonction backward()
        # pour que ce la fonctionne il faut spécifié que l'on veut que notre backpropagation se base sur les tensors w1 et w2
        # et cela en ajoutant un requires_grad=True dans la dénition de notre tensor
        loss.backward()

        #ici on update manuellement les poids en utilisant la descente du gradient. on wrap dans le torch.no_grad()
        # parce que on a le requires_grad=True dans le définition de nos poids
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # on remet les gradients à 0 après avoir update les poids,
            # on ne veut pas que les anciens gradients viennent perturber les nouveaux
            w1.grad.zero_()
            w2.grad.zero_()

def gradient_descent_pytorch():
    batch_size, input_dim, hidden_layer_size, output_dim = 64, 1000, 100, 10

    x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    y = torch.randn(batch_size, output_dim, device=device, dtype=dtype)

    w1 = torch.randn(input_dim, hidden_layer_size, device=device, dtype=dtype)
    w2 = torch.randn(hidden_layer_size, output_dim, device=device, dtype=dtype)

    for i in range(epochs):
        #comme pour numpy sauf que dot ici devient .mm()
        h = x.mm(w1)
        #comme pour numpy sauf que maximum ici devient .clamp(min=0)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        #vulgarisé ?
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
        #forward pass pour obtenir y_pred
        #x.dot(w1) permet de faire une multiplication de matrice entre le vecteur x (qui représente nos inputs)
        #et le vecteur w1 (qui représente les poids de nos inputs)
        h = x.dot(w1)
        #np.maximum(h, 0) on récupère une array contenant tout les éléments supérieur à 0
        h_relu = np.maximum(h, 0)
        #h_relu.dot(w2) multiplication de matrice entre vecteur h_relu(tous les éléments positif de la première mulpication matriciel entre x et w1)
        # et le vecteur w2 (qui représente les poids des outputs)
        y_pred = h_relu.dot(w2)
        #x->w1->relu->w2->y_pred

        #définition de la loss MSELoss: Mean Square Error Loss besoin de vulgariser ?
        #la somme de y_pred - y au carré
        #loss = np.square(y_pred - y).sum()

        #Back propagation en considération de la loss(ici on utilise pas la définition de la loss mais on la fait nous)
        #2(y_pred-y) equation MSELoss
        gradient_y_pred = 2.0 * (y_pred - y)
        #calcul du gradient de w2 avec h_relu avec le gradient du y_pred
        #ici le .T permet de transposer un vecteur 2 dimension
        gradient_w2 = h_relu.T.dot(gradient_y_pred)
        #calcul du gradient de h_relu avec le gradient du y_pred et w2
        gradient_h_relu = gradient_y_pred.dot(w2.T)
        #copie du gradient h relu pour le gradient_h
        gradient_h = gradient_h_relu.copy()
        #dans le gradient_h, tous ce qui est inférieur à 0 dans le h est mis à 0
        #lacune je comprend pas
        gradient_h[h < 0] = 0
        #calcul du gradient_w1 avec x et gradient_h
        gradient_w1 = x.T.dot(gradient_h)

        #update des poids en fonction du learning_rate trop mettre fait déborder l'apprentissage devient alors un surentrainement
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
            #représente notre model de prediction, formule mathématique
            prediction = m * X + b

            #l'erreur c'est la prediction moins l'output
            error = prediction - Y

            #on met à jour notre model en fonction du learning rate
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