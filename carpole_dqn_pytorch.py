import gym
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
from torch import nn
from torch.nn import functional
from torchvision import transforms
from torch import optim

#ici on utilise unwrapped car on veut ce qu'il se passe derrière
# dans l'environnement dymanique carpole
env = gym.make("CartPole-v0").unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ici on créer un tuple qui represente une seule transition dans notre environnement
# il map state action avec nex_state et reward
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

#le replay buffer c'est notre mémoire d'entrainement pour notre dqn.
# il enregistre les transitions que l'agent observe permettant d'y réutiliser plus tard
class replay_memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        #créer de l'espace
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        #rajoute la transition observée (*args récupère tout les paramètres)
        self.memory[self.position] = Transition(*args)
        self.position += 1

    #créer un batch random pour l'entrainement
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class dqn_agent(nn.Module):

    #réseau de neuronne convolutif, batchnorm2d permettra le calcul sur des réseau 3d
    def __init__(self, h, w, outputs):
        super(dqn_agent, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        #le nombre de linear input connection depend des output conv2d layers
        # et donc de la taille de l'image alors on calcul
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    #appelé avec un élément pour determiner notre prochaine action return un tensor
    def forward(self, x):
        #rectified linear sur le réseau
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


#_________________________________________________________________________________________________
#ici on extrait et process les images rendue par l'environnement.
# on utilise torchvision qui facilite le calcule des images
# directement récupéré depuis le site
resize = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize(40, interpolation=Image.CUBIC),
                    transforms.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)
#_______________________________________________________________________________________________________

env.reset()
#plt.figure()
#plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
#plt.title('Example extracted screen')
#plt.show()

#hyperparameters
batch_size = 128
gamma = 0.999
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 200
target_update = 10

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

#policy oriented
policy_net = dqn_agent(screen_height, screen_width, n_actions).to(device)
target_net = dqn_agent(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#using optim RMSprop
optimizer = optim.RMSprop(policy_net.parameters())
memory = replay_memory(10000)

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    epsilon_threshold = epsilon_end + (epsilon_start-epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
    steps_done += 1
    if sample > epsilon_threshold:
        with torch.no_grad():
            #ici return la meilleure action
            #récupère le max du policy_net(state) (max retourne 2 tensors le max et l'index ou
            # il a été trouvé nous on veut que le max) et le view permet de reshape le tensor
            return policy_net(state).max(1)[1].view(1,1)
    else:
        #ici return action random permettant l'apprentissage stochastique (exploration)
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('training')
    plt.xlabel('episode')
    plt.ylabel('duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())

    plt.pause(0.001)

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    #transpose le batch pour qu'il devienne une Transition d'un batch-array
    batch = Transition(*zip(*transitions))

    #on créer un tuple d'était qui ne soit pas final (celui après que la simulation finisse )de transition qui ne soit pas None
    #ici mask return des true ou false
    #non_final_next_state l'état en soit
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

    #récupère les infos nécessaire
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #j'ai pas bien compris comment mais en gros on récupère les données de deux tensor en prenant que la deuxième dim (le 1 dans gather())
    #calcule Q(s,a), le model calcule Q(s) et on select les actions qui auraient été prise pour chaque batch_state
    state_action_value = policy_net(state_batch).gather(1, action_batch)

    #calcul Q(s') pour tout les next_state. values expected pour les actions qui sont non-final
    # en choisissant leur meilleur reward grace à max(1)[0]
    #return un batch avec des valeur à 0 (0.)
    next_state_values = torch.zeros(batch_size, device=device)
    #ici on recup les next_value_stat ou dans le non_final_mask l'indice est à true
    # et récupère dans le target_net les valeur max en les détachant ainsi l'update des gradient n'aura pas d'impact sur eux
    next_state_values[non_final_mask] = target_net(non_final_next_state).max(1)[0].detach()

    #ici l'information pour la back prop des qvalues
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    #déf de la loss : ici huber loss, quand la loss est petite on performe MSELoss
    # si grosse loss on performe AbsoluteErrorLoss : à revoir cette loss
    #unsqueeze return un tensor the taille 1
    loss = functional.smooth_l1_loss(state_action_value, expected_state_action_values.unsqueeze(1))

    #on 0 les gradients car on ne veut pas que les anciens gradients perturbe les nouveaux
    optimizer.zero_grad()
    #back prop en fonction de lal oss
    loss.backward()

    #et on update les poids, ici controle sur les poids car param = w0,w1,...,wn
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

#TRAINING
num_episodes = 50
for i in range(num_episodes):
    #initlaise l'env et state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        #choisis l'action performé
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        #observer le nouvel état
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        #enregistre la transition dans la memoire
        memory.push(state, action, next_state, reward)

        #on passe au next_state
        state = next_state

        #on fait le step de l'opti sur le target network
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    #comme le veut dqn on update le target_model par rapport au policy_model
    if i % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("complete")
env.render()
env.close()
plt.show()