import itertools
import math
import random

import numpy as np
import torch

import torch.nn as nn
from torch.nn import functional as F

from src.dqn import DQN, ReplayMemory, Transition, resCNN, vggCNN, myCNN


class Policy(nn.Module):
    def __init__(self, inits, lr=4e-4,
                 type="vgg",
                 method="classification",
                 thresh=0.8,
                 non_local=False):
        super().__init__()

        self.wallet = 100
        self.method = method
        if self.method == "reinforcment":
            self.policy_net = DQN(inits, non_local, type)
            self.target_net = DQN(inits, non_local, type)
            self.target_net.eval()
            for cnn in self.target_net.cnns:
                cnn.eval()
            self.load_all_state_dict()
            self.target_net.eval()
            self.optimizer_policy = torch.optim.Adam(itertools.chain(*self.policy_net.get_parameters()), lr=lr)
            self.memory = ReplayMemory(4096)
            self.batch_size = 128
            self.gamma = 0
            self.eps_start = 0.9
            self.eps_end = 0.05
            self.eps_decay = 200
            self.target_update = 10
            self.steps_done = 0
        else:
            if type is "res":
                self.cnns = [resCNN(init, non_local=non_local) for init in inits]
            if type is "vgg":
                self.cnns = [vggCNN(init) for init in inits]
            if type is "myCNN":
                self.cnns = [myCNN(init) for init in inits]

            self.thresh = thresh
            self.optimizers = [torch.optim.Adam(cnn.parameters(), lr=lr) for cnn in self.cnns]
            self.losses = [torch.nn.CrossEntropyLoss() for _ in self.cnns]

    def select_action(self, state, eval=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold or eval:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state), dim=1)
        else:
            return torch.argmax(torch.rand(state.shape[0], 3), dim=1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s.unsqueeze(dim=0) for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat([state.unsqueeze(dim=0) for state in batch.state])
        action_batch = torch.cat([action.unsqueeze(dim=0) for action in batch.action]).unsqueeze(dim=1)
        reward_batch = torch.cat([reward.unsqueeze(dim=0) for reward in batch.reward]).unsqueeze(dim=1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask], _ = torch.max(self.target_net(non_final_next_states), dim=1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(dim=1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(dim=1), expected_state_action_values)

        # Optimize the model
        self.optimizer_policy.zero_grad()
        loss.backward(retain_graph=True)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_policy.step()

    def compute_rewards(self, actions, labels):
        rewards = torch.zeros(actions.shape[0], dtype=torch.float)
        for i in range(actions.shape[0]):
            if actions[i] == 2:
                rewards[i] = 0.5
            else:
                if actions[i] == labels[i]:
                    rewards[i] = 1.0

        return rewards

    def train_step(self, batch, labels):

        if self.method == "reinforcment":
            actions = self.select_action(batch)
            rewards = self.compute_rewards(actions, labels)

            for i in range(batch.shape[0] - 1):
                self.memory.push(batch[i], actions[i], batch[i + 1], rewards[i])
        else:
            for cnn, optimizer, loss in zip(self.cnns, self.optimizers, self.losses):
                optimizer.zero_grad()
                pred_labels = cnn(batch)
                loss_cnn = loss(pred_labels, labels)
                loss_cnn.backward()
                optimizer.step()

    def validation_step(self, batch, labels, metric):

        if self.method == "reinforcment":
            with torch.no_grad():
                results = self.select_action(batch, eval=True).tolist()
                answers = 0
                for i in range(len(results)):
                    if results[i] == 2:
                        results[i] = 0.5
                    else:
                        answers += 1

                if self.method == "reinforcment":
                    results = torch.tensor(results, dtype=torch.float)
        else:
            predictions = []
            for (cnn, loss) in zip(self.cnns, self.losses):
                pred_labels = F.softmax(cnn(batch), dim=1)
                predictions.append(torch.argmax(pred_labels, dim=1).type(torch.float32))
                means = torch.mean(torch.cat([prediction.unsqueeze(dim=1) for prediction in predictions], dim=1), dim=1)
                results = (((means >= self.thresh).type(torch.float32) - ((1 - means) >= self.thresh).type(
                    torch.float32)) + 1) / 2
                answers = 0
                for i in range(len(results)):
                    if results[i] != 0.5:
                        answers += 1

        hit_batch, profit = metric(results, labels)

        return hit_batch, answers, profit

    def _train(self):
        if self.method == "reinforcment":
            self.policy_net.train()
            for cnn in self.policy_net.cnns:
                cnn.train()
        else:
            for cnn in self.cnns:
                cnn.train()

    def _eval(self):
        if self.method == "reinforcment":
            self.policy_net.eval()
            for cnn in self.policy_net.cnns:
                cnn.eval()
        else:
            for cnn in self.cnns:
                cnn.eval()

    def load_all_state_dict(self):
        for i in range(len(self.policy_net.cnns)):
            self.target_net.cnns[i].load_state_dict(self.policy_net.cnns[i].state_dict())
        self.target_net.output.load_state_dict(self.policy_net.output.state_dict())
