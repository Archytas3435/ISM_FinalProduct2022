import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, image_pad, device, capacity_per_quantile, num_quantiles):
        self.num_quantiles = num_quantiles
        self.capacity = capacity_per_quantile * num_quantiles
        self.quantile_size = capacity_per_quantile
        self.device = device
        self.image_pad = image_pad
        self.observations = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)
        self.delta_V = np.empty((self.capacity, 1), dtype=np.float32)
        self.indices = self.delta_V.argpartition(np.arange(0, self.capacity, self.capacity//self.num_quantiles)[1:-1])
        self.idx = 0
        self.full = False
    def __len__(self):
        return self.capacity if self.full else self.idx
    def add(self, obs, action, reward, next_obs, done, done_no_max, delta_V):
        # add on if not full
        if not self.full:
            np.copyto(self.observations[self.idx], obs)
            np.copyto(self.actions[self.idx], action)
            np.copyto(self.rewards[self.idx], reward)
            np.copyto(self.next_observations[self.idx], next_obs)
            np.copyto(self.not_dones[self.idx], not done)
            np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
            np.copyto(self.delta_V[self.idx], delta_V)
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0
        # replace randomly if full
        else:
            n = np.random.randint(0, self.capacity)
            np.copyto(self.observations[n], obs)
            np.copyto(self.actions[n], action)
            np.copyto(self.rewards[n], reward)
            np.copyto(self.next_observations[n], next_obs)
            np.copyto(self.not_dones[n], not done)
            np.copyto(self.not_dones_no_max[n], not done_no_max)
            np.copyto(self.delta_V[n], delta_V)
    def resort(self):
        if self.full:
            self.indices = self.delta_V.argpartition(np.arange(0, self.capacity, self.capacity//self.num_quantiles)[1:-1])
            # self.observations = self.observations[indices]
            # self.next_observations = self.next_observations[indices]
            # self.actions = self.actions[indices]
            # self.rewards = self.rewards[indices]
            # self.not_dones = self.not_dones[indices]
            # self.not_dones_no_max = self.not_dones_no_max[indices]
        else:
            pass
    def revalue(self, QNetwork):
        for i in range(self.capacity if self.full else self.idx):
            self.delta_V[i] = QNetwork.get_value(self.next_observations[i]) - QNetwork.get_value(self.observations[i])
    def sample(self, batch_size, q_num):
        if self.full:
            indices = np.random.randint(
                self.quantile_size*q_num,
                self.quantile_size*(q_num+1),
                size=batch_size
            )
        else:
            indices = np.random.randint(
                0,
                self.idx,
                size=batch_size
            )
        observations = self.observations[self.indices][indices]
        next_observations = self.next_observations[self.indices][indices]
        observations = random_crop(observations, self.image_pad)
        next_observations = random_crop(next_observations, self.image_pad)
        observations = torch.as_tensor(observations, device=self.device).float()
        next_observations = torch.as_tensor(next_observations, device=self.device).float()
        actions = torch.as_tensor(self.actions[self.indices][indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[self.indices][indices], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[self.indices][indices], device=self.device)
        return observations, actions, rewards, next_observations, not_dones_no_max
    def sample_by_indices(self, start, end):
        indices = np.arange(start, end)
        observations = self.observations[self.indices][indices]
        next_observations = self.next_observations[self.indices][indices]
        observations = random_crop(observations, self.image_pad)
        next_observations = random_crop(next_observations, self.image_pad)
        observations = torch.as_tensor(observations, device=self.device).float()
        next_observations = torch.as_tensor(next_observations, device=self.device).float()
        actions = torch.as_tensor(self.actions[self.indices][indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[self.indices][indices], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[self.indices][indices], device=self.device)
        return observations, actions, rewards, next_observations, not_dones_no_max

def random_crop(images, image_pad, out=84):
    n, c, h, w = images.shape
    crop_max = h + (image_pad * 2) - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=images.dtype)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        img = np.pad(img, image_pad, mode='constant')[image_pad:-image_pad, :, :]
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def random_translate(images, size, return_random_indices=False, h1s=None, w1s=None):
    n, c, h, w = images.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=images.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, images, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_indices:  # So can do the same to another set of images.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs
