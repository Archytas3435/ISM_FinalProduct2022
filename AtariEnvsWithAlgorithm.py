# Imports
from bisect import bisect
from collections import deque, namedtuple
import gym
from itertools import count
import numpy as np
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Initializations
device = torch.device("cpu")
env = gym.make("ALE/Breakout-v5", obs_type="rgb", render_mode="rgb_array")
env.reset()

# Constants
SCREEN_HEIGHT = 210
SCREEN_WIDTH = 160
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 5
N_ACTIONS = env.action_space.n
N_QUANTILES = 5
N_EPISODES = 10000

# Replay Memory
Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "next_state",
        "reward"
    )
)
# need to add changes in value functions
class ReplayMemory(object):
    def __init__(self, capacity, num_quantiles=N_QUANTILES):
        self.memory = []# np.array([], dtype=[("transition", namedtuple), ("delta_V", float)])
        self.capacity = capacity
        self.num_quantiles = num_quantiles
    def push(self, delta_V, *args):
        self.memory.append((Transition(*args), delta_V))
        self.memory = sorted(self.memory, key=lambda x: x[1])# np.sort(self.memory, order="delta_V")
    def sample(self, batch_size, quantile_num):
        sample = random.sample(
            self.memory[(quantile_num*self.capacity//self.num_quantiles):((quantile_num+1)*self.capacity//self.num_quantiles)],
            batch_size
        )
        return [i[0] for i in sample]
    def __len__(self):
        return len(self.memory)

# DQN
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.LayerNorm(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.LayerNorm(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.LayerNorm(32)
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size-kernel_size)//stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# Training
policy_net = DQN(SCREEN_HEIGHT, SCREEN_WIDTH, N_ACTIONS).to(device)
target_net = DQN(SCREEN_HEIGHT, SCREEN_WIDTH, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(1000)

steps_completed = 0
def select_action(state):
    global steps_completed
    steps_completed += 1
    # no epsilon-greedy for now
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)

def optimize_model(state, action, next_state, reward):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE, random.randint(0, N_QUANTILES-1)) # need to fix to match spaced repetition pattern
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    memory.push(state, action, next_state, reward, expected_state_action_values - state_action_values)
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def render(num_tries, max_num_frames):
    env.reset()
    frames = []
    rewards = []
    for i in range(num_tries):
        for j in range(max_num_frames):
            action = select_action(state)
            observation, reward, done, _ = env.step(action.item())
            frames.append(observation)
            rewards.append(reward)
            if done:
                break
    # write gif to file

rewards = []
for i in range(N_EPISODES):
    env.reset()
    state = env.step(0)
    for t in count():
        action = select_action(state)
        next_state, reward, done, metadata = env.step(action.item())
        # reward = torch.tensor([reward], device=device)
        if done:
            rewards.append(reward)
            next_state = None
        optimize_model(state, action, next_state, reward)
        state = next_state
    if i%TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# env.step(1)
# for i in range(100):
#     observation, _, _, metadata = env.step(env.action_space.sample())
#     Image.fromarray(observation).save(f"test/{i}.jpg")