import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import config


class BirdBrain(nn.Module):
    # 5 inputs -> 128 -> 64 -> 2 outputs (Q values for flap / no-flap)
    def __init__(self, num_inputs=5, num_actions=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayMemory:
    """
    Stores past (state, action, reward, next_state, done) tuples.
    We sample random batches from this to break correlation between
    consecutive frames — this is what makes DQN training stable.
    """
    def __init__(self, capacity=config.MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent. Maintains two copies of BirdBrain:
      - policy_net:  actively trained every step
      - target_net:  frozen copy, synced periodically for stable Q-targets
    """
    def __init__(self):
        self.policy_net = BirdBrain()
        self.target_net = BirdBrain()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — less sensitive to outlier rewards than MSE
        self.memory = ReplayMemory()

        self.exploration_rate = config.EXPLORATION_START
        self.batch_size = config.BATCH_SIZE
        self.discount = config.DISCOUNT_FACTOR
        self.sync_every = config.TARGET_SYNC_EVERY
        self.total_steps = 0

    def pick_action(self, state):
        # Random exploration — biased heavily toward NOT flapping (~7% flap).
        # Optimal play flaps about once every 15 frames. 50/50 or even 20/80
        # makes the bird rocket into the ceiling every time.
        if random.random() < self.exploration_rate:
            return 1 if random.random() < config.RANDOM_FLAP_RATE else 0

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_net(state_t).argmax().item()

    def learn(self):
        # Don't learn until buffer has enough variety
        if len(self.memory) < config.WARMUP_STEPS:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q(s, a) from the policy net
        predicted_q = self.policy_net(states).gather(1, actions).squeeze()

        # max Q(s', a') from the target net (frozen)
        best_next_q = self.target_net(next_states).max(1)[0].detach()

        # Bellman target: r + gamma * max Q(s', a')
        target_q = rewards + (1 - dones) * self.discount * best_next_q

        loss = self.loss_fn(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), config.GRAD_CLIP)
        self.optimizer.step()

        return loss.item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1
        if self.total_steps % config.TRAIN_EVERY_N_FRAMES == 0:
            return self.learn()
        return None

    def decay_exploration(self):
        self.exploration_rate = max(config.EXPLORATION_MIN,
                                   self.exploration_rate * config.EXPLORATION_DECAY)

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.exploration_rate = config.EXPLORATION_MIN
