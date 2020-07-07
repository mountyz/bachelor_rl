import torch
import torch.nn as nn
import torch.nn.functional as F


def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    """Applies orthogonal initialization for the parameters of a given module"""

    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale

    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, constant_bias)


class DenseDQN(nn.Module):

    def __init__(self, env, with_layernorm=True, hidden_size=64):
        super(DenseDQN, self).__init__()

        # Extract dimensions from the environment
        ob_dim = env.observation_space.shape[0]  # state dimension
        ac_dim = env.action_space.n  # number of actions

        # Create fully-connected layers
        self.fc_1 = nn.Linear(ob_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        ortho_init(self.fc_1, nonlinearity='relu', constant_bias=0.0)
        ortho_init(self.fc_2, nonlinearity='relu', constant_bias=0.0)
        # Define layernorm layers
        self.ln_1 = nn.LayerNorm(hidden_size) if with_layernorm else lambda x: x
        self.ln_2 = nn.LayerNorm(hidden_size) if with_layernorm else lambda x: x

        # Define output head
        self.q_head = nn.Linear(hidden_size, ac_dim)
        ortho_init(self.q_head, nonlinearity='linear', constant_bias=0.0)

    def forward(self, ob):
        plop = ob
        # Stack fully-connected layers
        plop = F.relu(self.ln_1(self.fc_1(plop)))
        plop = F.relu(self.ln_2(self.fc_2(plop)))
        # Go through the output head
        q = self.q_head(plop)
        return q
