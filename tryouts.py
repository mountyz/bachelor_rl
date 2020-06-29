import numpy as np
import gym
import matplotlib as plt

learning_rate = 0.1
discount = 0.95
episodes = 5000
show = 200
env = gym.make("MountainCar-v0")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
start_epsilon_decay = 1
end_epsilon_decay = episodes // 2

epsilon_decay_value = epsilon//(end_epsilon_decay - start_epsilon_decay)

q_table = (np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])))

ep_reward = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for epochs in range(episodes):
    episode_reward = 0
    if epochs % show == 0:
        print(epochs)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        episode_reward += reward
        if render:
            print(action)
            stop = input("stop ? y")
            env.render()

        if not done:
            max_futur_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_futur_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state

    if end_epsilon_decay >= epochs >= start_epsilon_decay:
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

