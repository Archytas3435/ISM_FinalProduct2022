# Imports
import gym
import math
import random
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

env = gym.make("CartPole-v1").unwrapped
device = torch.device("cpu")

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
class ReplayMemory(object):
    def __init__(self, capacity, num_quantiles):
        self.memory = deque([], maxlen=capacity)
        self.num_quantiles = num_quantiles
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, quantile_num):
        sz = len(self.memory)//self.num_quantiles
        return self.memory[sz*quantile_num:sz*(quantile_num)+1]
    def __len__(self):
        return len(self.memory)

# DQN
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
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

# Interacting with the Environment
resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])
def get_cart_location(screen_width):
    return int(env.state[0]*screen_width/(env.x_threshold*2)+screen_width/2)
def get_screen():
    screen = env.render(mode="rgb_array").transpose((2, 0, 1)) # CHW
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height*0.8)]
    view_width = int(screen_width*0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width//2:
        slice_range = slice(view_width)
    elif cart_location > view_width//2:
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location-view_width//2, cart_location+view_width//2)
    screen = screen[:, :, slice_range]
    screen = torch.from_numpy(screen)
    screen = resize(screen).unsqueeze(0)
    return screen
env.reset()

# Training
BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 1000
TARGET_UPDATE = 5

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)

steps_completed = 0
def select_action(state):
    global steps_completed
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START-EPS_END) * math.exp(-1.0*steps_completed/EPS_DECAY)
    steps_completed += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
ma_len = 20
def plot_durations():
    plt.figure(0)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= ma_len:
        means = durations_t.unfold(0, ma_len, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(ma_len-1), means))
        plt.plot(means.numpy())
    plt.pause(0.001)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values*GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 1000
for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break
    if i_episode%TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.reset()
frames = []
for t in range(1000):
    frames.append(env.render(mode="rgb_array"))
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    action = select_action(state)
    _, reward, done, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    if done:
        env.reset()
env.close()
save_frames_as_gif(frames, filename="CartPole.gif")
