import gym
from utils import FrameStack, soft_update_params, _epsilon_schedule
from gym.wrappers import AtariPreprocessing
from replay_buffer_3 import ReplayBuffer
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Making the Environment"""
env = gym.make("ALE/Breakout-v5", obs_type="rgb", render_mode="rgb_array")

MAX_EP_STEPS = env._max_episode_steps
env = AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=False, noop_max=30)
env._max_episode_steps = MAX_EP_STEPS
env = FrameStack(env, k=4)
obs = env.reset()

NUM_QUANTILES = 5
QUANTILE_SIZE = 1000

"""Initializing the Replay Buffer"""
replay_buffer = ReplayBuffer(
	obs.shape,
	[1],
	4,
	DEVICE,
	QUANTILE_SIZE,
	NUM_QUANTILES
)

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
		self.fc = nn.Linear(3136, n_actions)
		self.n_actions = n_actions
		self.gamma = gamma
		self.device = device
		self.to(device)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
	def forward(self, observation):
		observation = observation / 255.
		out = self.conv_trunk(observation)
		out = out.view(out.size(0), -1)
		return self.fc(out)
	def update(self, replay_buffer, value_function_momentum, batch_size, q_num):
		observations, actions, rewards, next_observations, not_dones = replay_buffer.sample(
			batch_size,
			q_num
		)
		with torch.no_grad():
			momentum_target = value_function_momentum(next_observations)
			next_actions = momentum_target.argmax(1).reshape(batch_size, 1)
			momentum_target = momentum_target.gather(1, next_actions.long())
			momentum_target = rewards + self.gamma * momentum_target * not_dones
		q_sa = self.forward(obs).gather(1, actions.long())
		loss = F.mse_loss(momentum_target, q_sa)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
	def act(self, observation, epsilon):
		if np.random.rand() > epsilon:
			observation = torch.as_tensor(observation, device=self.device).float().unsqueeze(0)
			logits = self.forward(observation)
			return logits.argmax().cpu().item()
		else:
			return np.random.randint(0, self.n_actions)
	def get_value(self, observation):
		observation = torch.as_tensor(observation, device=self.device).float().unsqueeze(0)
		logits = self.forward(observation)
		return logits.max()

value_function = QNetwork(4, env.action_space.n, 0.99, DEVICE)
value_function_momentum = QNetwork(4, env.action_space.n, 0.99, DEVICE)
value_function_momentum.load_state_dict(value_function.state_dict())

"""Epsilon Schedule"""
epsilon_schedule = _epsilon_schedule(
	0.95,
	0.05,
	6.5,
	25000
)

"""Learning Pattern"""
pattern = [[j for j in range(i)] for i in range(1, NUM_QUANTILES+1)]
pattern = [i for j in pattern for i in j]
def get_quantile_num(current_ep, total_eps):
	return pattern[current_ep // total_eps]

"""Learning Loop"""
total_reward = []
eps_history = []
N_EPISODES = 1000
N_SEED_STEPS = 25000
SOFT_UPDATE_FREQ = 5
MEMORY_RESORT_FREQ = 10
BATCH_SIZE = 32
steps = 0
training_steps = 0

for i in tqdm(range(N_EPISODES)):
	observation = env.reset()
	done = False
	eps_reward = 0
	while not done:
		eps_history.append(epsilon_schedule(training_steps))
		action = value_function.act(observation, epsilon_schedule(training_steps))
		next_observation, reward, done, info = env.step(action)
		replay_buffer.add(obs, action, reward, next_observation, done, done, value_function.get_value(next_observation) - value_function.get_value(observation))
		eps_reward += reward
		steps += 1
		q_num = get_quantile_num(i, N_EPISODES)
		if steps >= N_SEED_STEPS:
			value_function.update(
				replay_buffer,
				value_function_momentum,
				BATCH_SIZE,
				q_num
			)
			training_steps += 1
		if steps % SOFT_UPDATE_FREQ == 0:
			soft_update_params(value_function, value_function_momentum, 0.05)
		if steps % MEMORY_RESORT_FREQ == 0:
			replay_buffer.revalue(value_function)
			replay_buffer.resort()
		observation = next_observation
	total_reward.append(eps_reward)

plt.plot(total_reward)
plt.show()

plt.plot(eps_history)
plt.show()
