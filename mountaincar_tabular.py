#EXERCICE REPRIS DES TUTORIELS PYTHON :
# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
import numpy as np
import gym
from matplotlib import pyplot as plt

learning_rate = 0.1
discount = 0.95
epochs = 5000
show = 200
env = gym.make("MountainCar-v0")

#la taille du tableau : 20xlen(env.obs_space.high)==20
discrete_obs_space_size = [20] * len(env.observation_space.high)
#taille des possibilité : 0.09 - 0.007
discrete_obs_space_win_size = (env.observation_space.high - env.observation_space.low) / discrete_obs_space_size

#pour l'exploration du bot
epsilon = 0.5
start_epsilon_decay = 1
#ici le double // permet le retour d'un int jamais de float
end_epsilon_decay = epochs // 2

epsilon_decay_value = epsilon / (end_epsilon_decay - start_epsilon_decay)

#on initialise les poids de la q_table avec des nombre entre 0 et -2 car, la meilleure récompense de
# l'environnement est de 0 (flag point) sinon c'est toujours une récompense négative
# et la taille est et de 20x20 comme expliqué en haut donc le maximum d'obersvation possible par le nombre
# maximum d'action donc 3 ce qui nous fait une taille de 20x20x3 donc une table 3 dimensionnel
q_table = (np.random.uniform(low=-2, high=0, size=(discrete_obs_space_size + [env.action_space.n])))

ep_reward = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

#on a besoin de récupérer les états mais ils sont continu donc il faut les rendre discret pour
# pouvoir travailler avc eux : pour ca on prend l'état en paramètre on le soustrait à l'obs_space.low et on le
# divise par la range entre le max et min puis on retourne un tuple
# (qui met les 2 nombre du discrete_state dans des parenthèse) puis on le transforme avec astype en np.int
# finalement nos résultats vascielleront entre (1,15) et (15,1) un peu près
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_obs_space_win_size
    return tuple(discrete_state.astype(np.int))

for epoch in range(epochs):
    episode_reward = 0
    #ici active le render
    if epoch % show == 0:
        print(epoch)
        render = True
    else:
        render = False
    #retourne le premier state
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        #choisis la meilleures actions définie dans la q_table
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        #transforme le new_state en discrete
        new_discrete_state = get_discrete_state(new_state)
        episode_reward += reward
        #render après récupération
        if render:
            env.render()
        if not done:
            #on prend la qvalue max dans la qtable pour la formule math
            # (pour faire la backprop en fonction la reward du state prime)
            max_futur_q = np.max(q_table[new_discrete_state])
            #récupère la current_q dans la qtable en fonction du discrete state et de l'action
            current_q = q_table[discrete_state + (action, )]
            #print(q_table[discrete_state][action], q_table[discrete_state + (action, )])
            #et ici on utilise la formule pour calculer la nouvelle qvalue comme décrit
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_futur_q)
            #update la q_table avec la new_q donc on prend dans la qtable en fonction
            # de discrete state et action et on met la new_q
            q_table[discrete_state + (action, )] = new_q
        #on définit notre récompense en cas de reussite
        # donc si new_state[0] position et velocity représnte le goal_position de l'env on set à 0
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
        #et on update le discrete_state avec le new_discrite_state
        discrete_state = new_discrete_state

    if end_epsilon_decay >= epoch >= start_epsilon_decay:
        epsilon -= epsilon_decay_value

    ep_reward.append(episode_reward)
    if not epochs % show:
        avg_reward = sum(ep_reward[-show:])/len(ep_reward[-show:])
        aggr_ep_rewards['ep'].append(epochs)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_reward[-show:]))
        aggr_ep_rewards['max'].append(max(ep_reward[-show:]))

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()

env.close()

