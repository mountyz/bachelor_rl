# Reinforcement Learning Exercise

In this assignment, we will use the
[DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
to train an agent in the [`CartPole-v1`](https://gym.openai.com/envs/CartPole-v1/)
environment,
provided via the [OpenAi Gym](https://github.com/openai/gym)
API.
The implementation is done in [PyTorch](https://pytorch.org/).

**Objective.** Complete the missing pieces to enable your agent to solve the task!

## Setup

**Instructions.** Sequentially execute the following commands:
```bash
# If you haven't done so already, install Anaconda/Miniconda on your system.
conda create -n pytorch-rl python=3.7
conda activate pytorch-rl
pip install --upgrade pip
pip install --upgrade pytest pytest-instafail wrapt six tqdm pyyaml psutil cloudpickle
pip install --upgrade numpy pandas scipy scikit-learn matplotlib
pip install --upgrade torch torchvision visdom
conda install -y -c conda-forge pillow pyglet pyopengl

cd && mkdir -p Code && cd Code
git clone https://github.com/openai/gym.git
pip install -e 'gym[classic_control]'
```

## How to launch an experiment?

* Launch a [Visdom]() instance with the command
```bash
visdom
```
* In a seperate terminal window, launch the experiment with the command
```bash
python main.py
```
You can set the hyperparameters (see the available ones in `argparsers.py`)
by adding options to the previous command:
```bash
python main.py --batch_size=64 --gamma=0.999
```
