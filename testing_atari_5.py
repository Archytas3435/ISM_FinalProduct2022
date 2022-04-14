import gym
from utils import FrameStack, soft_update_params, _epsilon_schedule
from gym.wrappers import AtariPreprocessing
from replay_buffer_5 import ReplayBuffer
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

"""Setting Constants"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on : {DEVICE}')
CAPACITY = 100000
parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=int)
args = parser.parse_args()

"""Making the Environment"""
env = gym.make("ALE/Breakout-v5", obs_type="rgb", render_mode="rgb_array")
MAX_EP_STEPS = env._max_episode_steps
env = AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=False, noop_max=30)
env._max_episode_steps = MAX_EP_STEPS
env = FrameStack(env, k=4)
obs = env.reset()

"""Value Function"""
class QNetwork(nn.Module):
	def __init__(self, n_channels, n_actions, gamma, device):
		super().__init__()
		self.conv_trunk = nn.Sequential(
			nn.Conv2d(n_channels, 32, 8, 4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, 2),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1),
			nn.ReLU()
		)
		self.fc = nn.Sequential(nn.Linear(3136, 256), nn.ReLU(), nn.Linear(256, n_actions))
		self.n_actions = n_actions
		self.gamma = gamma
		self.device = device
		self.to(device)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
	def forward(self, observation):
		observation = observation / 255.
		out = self.conv_trunk(observation)
		out = out.view(out.size(0), -1)
		return self.fc(out)
	def update(self, replay_buffer, value_function_momentum, num):
		observations, actions, rewards, next_observations, not_dones, batch_size = replay_buffer.sample(
			num
		)
		with torch.no_grad():
			momentum_target = value_function_momentum(next_observations)
			next_actions = momentum_target.argmax(1).reshape(batch_size, 1)
			momentum_target = momentum_target.gather(1, next_actions.long())
			momentum_target = rewards + self.gamma * momentum_target * not_dones
		q_sa = self.forward(observations).gather(1, actions.long())
		loss = F.mse_loss(momentum_target, q_sa)
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()
	def update_old(self, replay_buffer, value_function_momentum, num):
		observations, actions, rewards, next_observations, not_dones, batch_size = replay_buffer.sample_old(
			num
		)
		with torch.no_grad():
			momentum_target = value_function_momentum(next_observations)
			next_actions = momentum_target.argmax(1).reshape(batch_size, 1)
			momentum_target = momentum_target.gather(1, next_actions.long())
			momentum_target = rewards + self.gamma * momentum_target * not_dones
		q_sa = self.forward(observations).gather(1, actions.long())
		loss = F.mse_loss(momentum_target, q_sa)
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()
	def act(self, observation, epsilon):
		if np.random.rand() > epsilon:
			observation = torch.as_tensor(observation, device=self.device).float().unsqueeze(0)
			logits = self.forward(observation)
			return logits.argmax().cpu().item()
		else:
			return np.random.randint(0, self.n_actions)
	def get_greedy_value(self, observation):
		with torch.no_grad():
			observation = torch.as_tensor(observation, device=self.device).float().unsqueeze(0)
			logits = self.forward(observation)
		return logits.max()
	def get_greedy_value_vec(self, observation):
		with torch.no_grad():
			observation = torch.as_tensor(observation, device=self.device).float()
			logits = self.forward(observation)
		return logits.max(dim=-1, keepdim=True)[0]
	def get_value(self, observation, action):
		with torch.no_grad():
			observation = torch.as_tensor(observation, device=self.device).float().unsqueeze(0)
			logits = self.forward(observation)
		return logits[0][action]
	def get_value_vec(self, observation, action):
		with torch.no_grad():
			observation = torch.as_tensor(observation, device=self.device).float()
			logits = self.forward(observation)
		return logits.gather(1, torch.as_tensor(action, device=self.device).long())

value_function = QNetwork(4, env.action_space.n, 0.99, DEVICE)
value_function_momentum = QNetwork(4, env.action_space.n, 0.99, DEVICE)
value_function_momentum.load_state_dict(value_function.state_dict())

"""Initializing the Replay Buffer"""
replay_buffer = ReplayBuffer(
	obs.shape,
	[1],
	4,
	DEVICE,
	CAPACITY
)

"""Epsilon Schedule"""
epsilon_schedule = _epsilon_schedule(
	0.95,
	0.05,
	1.5,
	50000
)

action_hist = []

"""Learning Loop"""
total_reward = []
eps_history = []
N_EPISODES = 2000
N_SEED_STEPS = 50000
SOFT_UPDATE_FREQ = 5
MEMORY_RESORT_FREQ = 10
BATCH_SIZE = 32
steps = 0
training_steps = 0
training_episodes = 0
BEGIN_LS = 1000
initial_shuffle_done = False

for i in tqdm(range(N_EPISODES)):
	observation = env.reset()
	done = False
	eps_reward = 0
	collect_actions = False
	while not done:
		eps_history.append(epsilon_schedule(training_steps))
		action = value_function.act(observation, epsilon_schedule(training_steps))
		next_observation, reward, done, info = env.step(action)
		replay_buffer.add(
			observation,
			action,
			reward,
			next_observation,
			done,
			done,
			(reward + value_function_momentum.gamma * value_function_momentum.get_greedy_value(next_observation) - value_function.get_value(observation, action)).abs()
		)
		eps_reward += reward
		steps += 1
		if steps >= N_SEED_STEPS:
			if i >= BEGIN_LS:
				num = i%10
				value_function.update(
					replay_buffer,
					value_function_momentum,
					num
				)
			else:
				value_function.update_old(
					replay_buffer,
					value_function_momentum,
					None
				)
			training_steps += 1
		if steps % SOFT_UPDATE_FREQ == 0:
			soft_update_params(value_function, value_function_momentum, 0.05)
		observation = next_observation
	if i >= N_EPISODES - BEGIN_LS:
		training_episodes += 1
	total_reward.append(eps_reward)
	if i % 100 == 0:
		print(f'Mean: {np.mean(total_reward[-100:])}, Eps: {epsilon_schedule(training_steps)}, Steps: {steps}')
