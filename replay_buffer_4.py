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
        # self.delta_V = np.empty((1, self.capacity), dtype=np.float32)

        # Not call right away?
        # self.indices = self.delta_V.argpartition(np.arange(0, self.capacity, self.capacity//self.num_quantiles)[1:-1])
        self.idx = 0
        self.full = False

        self.action_hist = []
        self.action_hist_sat = []

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
            np.copyto(self.delta_V[self.idx], delta_V.detach().cpu().numpy())
            # np.copyto(self.delta_V[self.idx], delta_V.detach().cpu().numpy())
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
            np.copyto(self.delta_V[n], delta_V.detach().cpu().numpy())
            # np.copyto(self.delta_V[0, self.idx], delta_V.detach().cpu().numpy())

    def resort(self):
        if self.full:
            self.indices = self.delta_V.argsort(axis=0)[::-1]

        else:
            pass

    def revalue(self, QNetwork, QNetwork_target):
        b_idx = 0
        e_idx = 1

        for i in range(int(self.capacity / 500)):
            with torch.no_grad():
                # torch.as_tensor(self.rewards[500*b_idx:500*e_idx], device=self.device).float() \
                target = torch.as_tensor(self.rewards[500*b_idx:500*e_idx], device=self.device).float() \
                         + (QNetwork_target.gamma \
                         * QNetwork_target.get_greedy_value_vec(self.next_observations[500*b_idx:500*e_idx]) \
                         * torch.as_tensor(self.not_dones_no_max[500*b_idx:500*e_idx], device=self.device).float())

                val_diff = target \
                           - QNetwork.get_value_vec(self.observations[500*b_idx:500*e_idx],
                                                self.actions[500*b_idx:500*e_idx])

            val_diff = val_diff.abs().cpu().numpy()

            for val_idx, j in enumerate(range(500*b_idx, 500*e_idx)):
                np.copyto(self.delta_V[j], val_diff[val_idx])

            b_idx += 1
            e_idx += 1

    def sample_old(self, batch_size, *args):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        # if np.random.rand() > 0.99:
        #     print(idxs)

        obses = self.observations[idxs]
        next_obses = self.next_observations[idxs]
        # obses_aug = obses.copy()
        # next_obses_aug = next_obses.copy()

        # Augmentation
        obses = random_crop(obses, self.image_pad)
        next_obses = random_crop(next_obses, self.image_pad)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        # obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        # next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        # KEEP UNCOMMENTED FOR TRANSLATION
        # For ablations with no augmentation, *comment out* the following lines
        # obses = self.aug_trans(obses)
        # next_obses = self.aug_trans(next_obses)

        # obses_aug = self.aug_trans(obses_aug)
        # next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max  #, obses_aug, next_obses_aug

    def generate_placeholder(self):
        self.indices = np.arange(0, self.capacity).reshape(self.capacity, 1)

    def sample(self, batch_size, q_num):
        # if self.full:
        inds = np.random.randint(
            self.quantile_size*q_num,
            self.quantile_size*(q_num+1),
            size=batch_size
        )

        self.action_hist.extend(inds.tolist())

        inds = self.indices[inds][:, 0]

        self.action_hist_sat.extend(inds.tolist())

        observations = self.observations[inds]#[:, 0, :, :, :]
        next_observations = self.next_observations[inds]#[:, 0, :, :, :]

        observations = random_crop(observations, self.image_pad)
        next_observations = random_crop(next_observations, self.image_pad)

        observations = torch.as_tensor(observations, device=self.device).float()
        next_observations = torch.as_tensor(next_observations, device=self.device).float()
        # actions = torch.as_tensor(self.actions[self.indices][indices][:, 0, :], device=self.device)
        actions = torch.as_tensor(self.actions[inds], device=self.device)
        rewards = torch.as_tensor(self.rewards[inds], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[inds], device=self.device)
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