import gym
from utils import FrameStack, soft_update_params, _epsilon_schedule
from gym.wrappers import AtariPreprocessing
from replay_buffer_2 import ReplayBuffer
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Making the Environment"""
env = gym.make("ALE/Breakout-v5", obs_type="rgb", render_mode="rgb_array")


MAX_EP_STEPS = env._max_episode_steps
env = AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=False, noop_max=30)
env._max_episode_steps = MAX_EP_STEPS
env = FrameStack(env, k=4)
obs = env.reset()
import matplotlib.pyplot as plt

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

	def forward(self, obs):
		obs = obs / 255.
		out = self.conv_trunk(obs)
		out = out.view(out.size(0), -1)
		return self.fc(out)

	def update(self, replay_buffer, value_function_momentum, batch_size, q_num):
		obs, actions, rewards, next_obses, not_dones = replay_buffer.sample(
			batch_size
		)

		with torch.no_grad():
			momentum_target = value_function_momentum(next_obses)
			next_actions = momentum_target.argmax(1).reshape(batch_size, 1)
			momentum_target = momentum_target.gather(1, next_actions.long())
			momentum_target = rewards + self.gamma * momentum_target * not_dones

		q_sa = self.forward(obs).gather(1, actions.long())

		loss = F.mse_loss(momentum_target, q_sa)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def act(self, obs, epsilon):
		if np.random.rand() > epsilon:
			obs = torch.as_tensor(obs, device=self.device).float()
			obs = obs.unsqueeze(0)
			logits = self.forward(obs)
			return logits.argmax().cpu().item()

		else:
			return np.random.randint(0, self.n_actions)


value_function = QNetwork(4, env.action_space.n, 0.99, DEVICE)
value_function_momentum = QNetwork(4, env.action_space.n, 0.99, DEVICE)
value_function_momentum.load_state_dict(value_function.state_dict())

"""Epsilon Schedule"""
eps_sched = _epsilon_schedule(
	0.95, 0.05, 6.5, 25000
)

def _plot_epsilon(eps_sched, total_steps):
	x = np.arange(0, 1, 0.001) * total_steps
	y = list(map(eps_sched, x))
	plt.plot(x, y)
	plt.show()

# _plot_epsilon(eps_sched, 100000)

"""Learning Loop"""
total_reward = []
eps_history = []
N_EPISODES = 1000
N_SEED_STEPS = 25000
SOFT_UPDATE_FREQ = 5
BATCH_SIZE = 32
steps = 0
training_steps = 0

pattern = [[j for j in range(i)] for i in range(1, NUM_QUANTILES+1)]
pattern = [i for j in pattern for i in j] # efficiency

def get_quantile_num(current_step, total_steps):
	return pattern[current_step // total_steps]

for i in tqdm(range(N_EPISODES)):
	obs = env.reset()
	done = False
	eps_reward = 0

	while not done:
		eps_history.append(eps_sched(training_steps))

		action = value_function.act(obs, eps_sched(training_steps))
		next_obs, reward, done, info = env.step(action) # how does this calculate reward

		replay_buffer.add(obs, action, reward, next_obs, done, done)

		eps_reward += reward

		steps += 1

		q_num = get_quantile_num(steps, N_SEED_STEPS) # ?

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

		obs = next_obs

	total_reward.append(eps_reward)

plt.plot(total_reward)
plt.show()

plt.plot(eps_history)
plt.show()