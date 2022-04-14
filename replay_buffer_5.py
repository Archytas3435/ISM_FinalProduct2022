import numpy as np
import torch

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

class InitialBox:
    def __init__(self, obs_shape, action_shape, image_pad, device, capacity):
        self.capacity = capacity
        self.device = device
        self.image_pad = image_pad
        self.observations = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)
        self.metric_val = np.empty((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.action_history = []
        self.action_history_sat = []
        self.indices = np.arange(0, 1)
    def __len__(self):
        return self.capacity if self.full else self.idx
    def add(self, obs, action, reward, next_obs, done, done_no_max, metric_val):
        if not self.full:
            np.copyto(self.observations[self.idx], obs)
            np.copyto(self.actions[self.idx], action)
            np.copyto(self.rewards[self.idx], reward)
            np.copyto(self.next_observations[self.idx], next_obs)
            np.copyto(self.not_dones[self.idx], not done)
            np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
            np.copyto(self.metric_val[self.idx], metric_val.detach().cpu().numpy())
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0
        else:
            n = np.random.randint(0, self.capacity)
            np.copyto(self.observations[n], obs)
            np.copyto(self.actions[n], action)
            np.copyto(self.rewards[n], reward)
            np.copyto(self.next_observations[n], next_obs)
            np.copyto(self.not_dones[n], not done)
            np.copyto(self.not_dones_no_max[n], not done_no_max)
            np.copyto(self.metric_val[n], metric_val.detach().cpu().numpy())
    def sample(self):
        self.action_history.extend(indices.tolist())
        self.action_history_sat.extend(indices.tolist())
        observations = self.observations[indices]
        next_observations = self.next_observations[indices]
        observations = random_crop(observations, self.image_pad)
        next_observations = random_crop(next_observations, self.image_pad)
        observations = torch.as_tensor(observations, device=self.device).float()
        next_observations = torch.as_tensor(next_observations, device=self.device).float()
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[indices], device=self.device)
        return observations, actions, rewards, next_observations, not_dones_no_max, len(indices)
    def remove_element(self, index):
        # Needs to be checked
        a = (self.observations[index], self.actions[index], self.rewards[index], self.next_observations[index],
             not self.not_dones[index], not self.not_dones_no_max[index], self.metric_val[index])
        np.copyto(self.observations[index], self.observations[self.idx])
        np.copyto(self.actions[index], self.actions[self.idx])
        np.copyto(self.rewards[index], self.rewards[self.idx])
        np.copyto(self.next_observations[index], self.next_observations[self.idx])
        np.copyto(self.not_dones[index], self.not_dones[self.idx])
        np.copyto(self.not_dones_no_max[index], self.not_dones_no_max[self.idx])
        np.copyto(self.metric_val[index], self.metric_val[self.idx])
        self.idx = self.idx - 1
        return a

class OtherBox:
    def __init__(self, obs_shape, action_shape, image_pad, device, capacity, relevant_episodes):
        self.capacity = capacity
        self.device = device
        self.image_pad = image_pad
        self.observations = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.empty((self.capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)
        self.metric_val = np.empty((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.action_history = []
        self.action_history_sat = []
        self.relevant_episodes = relevant_episodes
    def __len__(self):
        return self.capacity if self.full else self.idx
    def add(self, obs, action, reward, next_obs, done, done_no_max, metric_val):
        if not self.full:
            np.copyto(self.observations[self.idx], obs)
            np.copyto(self.actions[self.idx], action)
            np.copyto(self.rewards[self.idx], reward)
            np.copyto(self.next_observations[self.idx], next_obs)
            np.copyto(self.not_dones[self.idx], not done)
            np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
            np.copyto(self.metric_val[self.idx], metric_val.detach().cpu().numpy())
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0
        else:
            n = np.random.randint(0, self.capacity)
            np.copyto(self.observations[n], obs)
            np.copyto(self.actions[n], action)
            np.copyto(self.rewards[n], reward)
            np.copyto(self.next_observations[n], next_obs)
            np.copyto(self.not_dones[n], not done)
            np.copyto(self.not_dones_no_max[n], not done_no_max)
            np.copyto(self.metric_val[n], metric_val.detach().cpu().numpy())
    def sample(self):
        if not self.full:
            indices = np.arange(0, self.idx)
        else:
            indices = np.arange(0, self.capacity)
        self.action_history.extend(indices.tolist())
        self.action_history_sat.extend(indices.tolist())
        observations = self.observations[indices]
        next_observations = self.next_observations[indices]
        observations = random_crop(observations, self.image_pad)
        next_observations = random_crop(next_observations, self.image_pad)
        observations = torch.as_tensor(observations, device=self.device).float()
        next_observations = torch.as_tensor(next_observations, device=self.device).float()
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[indices], device=self.device)
        return observations, actions, rewards, next_observations, not_dones_no_max, len(indices)
    def remove_element(self, index):
        # Needs to be checked
        a = (self.observations[index], self.actions[index], self.rewards[index], self.next_observations[index], not self.not_dones[index], not self.not_dones_no_max[index], self.metric_val[index])
        np.copyto(self.observations[index], self.observations[self.idx])
        np.copyto(self.actions[index], self.actions[self.idx])
        np.copyto(self.rewards[index], self.rewards[self.idx])
        np.copyto(self.next_observations[index], self.next_observations[self.idx])
        np.copyto(self.not_dones[index], self.not_dones[self.idx])
        np.copyto(self.not_dones_no_max[index], self.not_dones_no_max[self.idx])
        np.copyto(self.metric_val[index], self.metric_val[self.idx])
        self.idx = self.idx - 1
        return a

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, image_pad, device, capacity):
        self.box_0 = InitialBox(obs_shape, action_shape, image_pad, device, capacity)
        self.box_1 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [0, 2, 5, 9])
        self.box_2 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [1, 3, 6, 0])
        self.box_3 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [2, 4, 7, 1])
        self.box_4 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [3, 5, 8, 2])
        self.box_5 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [4, 6, 9, 3])
        self.box_6 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [5, 7, 0, 4])
        self.box_7 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [6, 8, 1, 5])
        self.box_8 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [7, 9, 2, 6])
        self.box_9 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [8, 0, 3, 7])
        self.box_10 = OtherBox(obs_shape, action_shape, image_pad, device, capacity//10, [9, 1, 4, 8])
        self.boxes = [self.box_1, self.box_2, self.box_3, self.box_4, self.box_5, self.box_6, self.box_7, self.box_8, self.box_9, self.box_10]
    def add(self, obs, action, reward, next_obs, done, done_no_max, metric_val):
        self.box_0.add(obs, action, reward, next_obs, done, done_no_max, metric_val)
    def get_identity_box(num):
        for i in self.boxes:
            if i.relevant_episodes[0] == num:
                return i
        return random.choice(self.boxes)
    def get_relevant_boxes(num):
        return [i for i in self.boxes if num in i.relevant_episodes]
    def process(self, relevant_box_num):
        """
        - for sample in box_0, if TD error is reduced, move to relevant box
        - learn from all boxes with relevant_box as one of the relevant values
            - if sample does not improve, move back to box_0
            - for all boxes with relevant_box as last relevant value, remove sample if TD error is reduced
        """
        relevant_box = get_identity_box(relevant_box_num)
        all_relevant_boxes = get_relevant_boxes(relevant_box_num)
        """Move box_0 samples"""
        b_idx = 0
        e_idx = 1
        for i in range(int(self.box_0.capacity / 500)):
            with torch.no_grad():
                target = (
                        torch.as_tensor(self.box_0.rewards[500 * b_idx:500 * e_idx], device=self.device).float()
                        + (
                                QNetwork_target.gamma
                                * QNetwork_target.get_greedy_value_vec(self.box_0.next_observations[500 * b_idx:500 * e_idx])
                                * torch.as_tensor(self.box_0.not_dones_no_max[500 * b_idx:500 * e_idx], device=self.device).float()
                        )
                )
                val_diff = target - QNetwork.get_value_vec(
                    self.box_0.observations[500 * b_idx:500 * e_idx],
                    self.box_0.actions[500 * b_idx:500 * e_idx]
                )
            val_diff = val_diff.abs().cpu().numpy()
            for val_idx, j in enumerate(range(500 * b_idx, 500 * e_idx)):
                if val_diff[val_idx] < self.box_0.metric[j]:
                    a = self.box_0.remove_element(j)
                    relevant_box.add(*a)
            b_idx += 1
            e_idx += 1
        """Move samples from other relevant boxes"""
        for box in all_relevant_boxes:
            b_idx = 0
            e_idx = 1
            for i in range(int(box.capacity / 500)):
                with torch.no_grad():
                    target = (
                            torch.as_tensor(box.rewards[500 * b_idx:500 * e_idx], device=self.device).float()
                            + (
                                    QNetwork_target.gamma
                                    * QNetwork_target.get_greedy_value_vec(box.next_observations[500 * b_idx:500 * e_idx])
                                    * torch.as_tensor(box.not_dones_no_max[500 * b_idx:500 * e_idx],
                                                      device=self.device).float()
                            )
                    )
                    val_diff = target - QNetwork.get_value_vec(
                        box.observations[500 * b_idx:500 * e_idx],
                        box.actions[500 * b_idx:500 * e_idx]
                    )
                val_diff = val_diff.abs().cpu().numpy()
                for val_idx, j in enumerate(range(500 * b_idx, 500 * e_idx)):
                    if val_diff[val_idx] > box.metric[j]:
                        a = box.remove_element(j)
                        self.box_0.add(*a)
                    elif val_diff[val_idx] < box.metric[j] and box.relevant_episodes[-1]==relevant_box_num:
                        box.remove_element(j)
                b_idx += 1
                e_idx += 1
    def sample(self, relevant_box):
        return self.boxes[relevant_box].sample()
