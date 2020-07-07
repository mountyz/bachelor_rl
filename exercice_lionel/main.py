import math
import random
from copy import copy
from collections import namedtuple, deque
import visdom
import torch
import torch.nn.functional as F
import torch.nn.utils as U
import numpy as np

import gym

from argparsers import argparser
from replay_buffer import ReplayBuffer
from networks import DenseDQN


class DQN(object):

    def __init__(self, env, device, hps):
        # Define attributes
        self.env = env
        self.device = device
        self.hps = hps
        # Create the online Q network (effectively, the policy)
        self.online_q_net = DenseDQN(self.env, self.hps.with_layernorm).to(self.device)
        # Create the target network, load the online net's weights in it, and put in eval mode
        self.target_q_net = DenseDQN(self.env, self.hps.with_layernorm).to(self.device)
        self.target_q_net.load_state_dict(self.online_q_net.state_dict())
        self.target_q_net.eval()
        # Create the optimizer
        self.optimizer = torch.optim.Adam(self.online_q_net.parameters(), lr=self.hps.lr)
        # Create the replay buffer
        self.memory = ReplayBuffer(capacity=self.hps.memory_size)
        # Define epsilon (epsilon-greedy exploration)
        self.eps = self.hps.eps_beg


    def update_target_net(self):
        self.target_q_net.load_state_dict(self.online_q_net.state_dict())

    def act(self, state, timesteps_so_far):
        # Convert the state from a numpy to tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Compute epsilon threshold choosing if the agent is going to explore or to exploit
        eps_threshold = self.hps.eps_end + (self.hps.eps_beg - self.hps.eps_end) * math.exp(-1 * timesteps_so_far / self.hps.eps_decay)
        if random.random() > eps_threshold:
            return self.online_q_net(state).max(1)[1].view(1, 1)
        else:
            return random.choice(np.arange(self.env.action_space.n))

    def collect(self):
        # Initialize every variable involved in the data collection
        timesteps_so_far = 0
        done = True
        env_rew = 0.0
        ob = self.env.reset()
        cur_ep_len = 0
        cur_ep_ret = 0
        ep_lens = []
        ep_rets = []
        while True:
            # Predict an action in the current state with the model
            ac = self.act(ob, timesteps_so_far)
            ac = ac if isinstance(ac, int) else ac.item()
            if timesteps_so_far > 0 and timesteps_so_far % self.hps.rollout_len == 0:
                # Return (yield) the collected data
                yield (ep_lens, ep_rets)
                # Once back in, reset the lists containing the episode statistics
                ep_lens = []
                ep_rets = []
            # Interact with the environment
            next_ob, rew, done, _ = self.env.step(ac)
            if self.hps.render:
                self.env.render()
            # Update episode statistics
            cur_ep_len += 1
            cur_ep_ret += rew
            # Store the experienced transition in the replay buffer
            self.memory.store({'ob': ob,
                               'ac': ac,
                               'rew': rew,
                               'next_ob': next_ob,
                               'done': done})
            # Make the next observation the current one
            ob = copy(next_ob)
            # If the episode is over, reset the environment and episode statistics
            if done:
                ob = self.env.reset()
                ep_lens.append(cur_ep_len)
                ep_rets.append(cur_ep_ret)
                cur_ep_len = 0
                cur_ep_ret = 0
            # Increment the interaction count
            timesteps_so_far += 1

    def train(self):
        # Get the random sample of transitions that the agent has observed
        transitions = self.memory.sample(args.batch_size)
        # Translate it to perform the optimization of the model
        states = torch.from_numpy(transitions['ob']).float().to(device)
        actions = torch.from_numpy(transitions['ac'].T).long().to(device)
        rewards = torch.from_numpy(transitions['rew'].T).float().to(device)
        next_states = torch.from_numpy(transitions['next_ob']).float().to(device)
        done = torch.from_numpy(transitions['done'].T.astype(np.uint8)).float().to(device)

        # Get max predicted Q values (for next states) from target model
        next_qvalues_target = self.target_q_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        qvalues_target = rewards + (args.gamma * next_qvalues_target * (1 - done))
        # Get expected Q values from local model
        expected_qvalues_online = self.online_q_net(states).gather(1, actions.unsqueeze(1))
        # Compute the loss
        loss = F.smooth_l1_loss(expected_qvalues_online, qvalues_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.online_q_net.parameters():
                param -= args.learning_rate * param.grad
        self.optimizer.step()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

args = argparser().parse_args()

# Set device-related knobs
if args.cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("device in use: {}".format(device))

# Create the environment and print its specs
env = gym.make('CartPole-v1')
print(">>>>>>>>>> ob_space: {} | ac_space: {}".format(env.observation_space,
                                                      env.action_space))
# Initialize the env's seed
env.seed(args.seed)
# Set the seed for all the sources of randomness
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Create Visdom instance
viz = visdom.Visdom()
ep_rets_win = viz.line(X=[0], Y=[np.nan])

# Create the DQN agent
agent = DQN(env, device, args)

# Create a generator to simulate interactions with the environment
interaction = agent.collect()

iters_so_far = 0
ep_lens_buffer = deque(maxlen=600)
ep_rets_buffer = deque(maxlen=600)
show = 200

for i in range(args.num_iters):



    # Collect samples by interacting with the environment
    ep_lens, ep_rets = interaction.__next__()
    ep_lens_buffer.extend(ep_lens)
    ep_rets_buffer.extend(ep_rets)

    # Optimize the Q network
    agent.train()

    if iters_so_far % args.target_update == 0:
        # Update the target network
        agent.update_target_net()

    iters_so_far += 1
    env.render()

    if iters_so_far % 100 == 0:

        # Print episode statistics
        viz.line(X=[iters_so_far],
                 Y=[np.nanmean(ep_rets_buffer)],
                 win=ep_rets_win,
                 update='append',
                 opts=dict(title='Episodic Return'))


