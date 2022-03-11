import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, image_pad, device, capacity_per_quantile, num_quantiles):
        self.num_quantiles = num_quantiles
        self.quantiles = [
            Quantile(obs_shape, action_shape, capacity_per_quantile, image_pad, device)
            for i in range(num_quantiles)
        ]
        self.thresholds = [quantile.get_min_max() for quantile in self.quantiles]
    def _update_thresholds(self):
        self.thresholds = [quantile.get_min_max() for quantile in self.quantiles]
    def __len__(self):
        return sum(len(quantile) for quantile in self.quantiles)
    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self._update_thresholds()
        q_num = self.num_quantiles - 1
        for i in range(self.num_quantiles):
            if reward < self.thresholds[i][1]:
                q_num = i
                break
        self.quantiles[q_num].add(obs, action, reward, next_obs, done, done_no_max)
    def sample(self, batch_size, q_num):
        return self.quantiles[q_num].sample(batch_size)

class Quantile:
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device
        # self.env = env
        self.image_pad = image_pad

        # self.aug_trans = nn.Sequential(
        #     nn.ReplicationPad2d(image_pad),
        #     kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1]))
        # )

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

        self.min = float("inf") # check max and min values
        self.max = float("-inf")

    def __len__(self):
        return self.capacity if self.full else self.idx
    def get_min_max(self):
        return self.min, self.max
    def add(self, obs, action, reward, next_obs, done, done_no_max):
        if reward > self.max:
            self.max = reward
        elif reward < self.min:
            self.min = reward
        if not self.full:
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.actions[self.idx], action)
            np.copyto(self.rewards[self.idx], reward)
            np.copyto(self.next_obses[self.idx], next_obs)
            np.copyto(self.not_dones[self.idx], not done)
            np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0
        else:
            n = np.random.randint(0, self.capacity)
            np.copyto(self.obses[n], obs)
            np.copyto(self.actions[n], action)
            np.copyto(self.rewards[n], reward)
            np.copyto(self.next_obses[n], next_obs)
            np.copyto(self.not_dones[n], not done)
            np.copyto(self.not_dones_no_max[n], not done_no_max)

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
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

def random_crop(imgs, image_pad, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h + (image_pad * 2) - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        img = np.pad(img, image_pad, mode='constant')[image_pad:-image_pad, :, :]
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs
