import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from torch import manual_seed
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class PPO:
    def __init__(self, numberOfInputs, numberOfActorOutputs, clip_param=0.2, max_grad_norm=0.5, ppo_update_iters=5,
                 batch_size=8, gamma=0.99, use_cuda=False, actor_lr=0.001, critic_lr=0.003, seed=None):
        super().__init__()
        if seed is not None:
            manual_seed(seed)

        # Hyper-parameters
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_iters = ppo_update_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_cuda = use_cuda

        # models
        self.actor_net = Actor(numberOfInputs, numberOfActorOutputs)
        self.critic_net = Critic(numberOfInputs)
        
        self.load('')
        if self.use_cuda:
            #print("yes")
            print(torch.cuda.is_available())
            self.actor_net.cuda()
            self.critic_net.cuda()

        # Create the optimizers
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), critic_lr)

        # Training stats
        self.buffer = []

    def work(self, agentInput, type_="simple"):
     
        agentInput = from_numpy(np.array(agentInput)).float().unsqueeze(0)
        if self.use_cuda:
            agentInput = agentInput.cuda()
        with no_grad():
            action_prob = self.actor_net(agentInput)

        if type_ == "simple":
            output = [action_prob[0][i].data.tolist() for i in range(len(action_prob[0]))]
            return output
        elif type_ == "selectAction":
            c = Categorical(action_prob)
            action = c.sample()
            return action.item(), action_prob[:, action.item()].item()
        elif type_ == "selectActionMax":
            return np.argmax(action_prob).item(), 1.0
        else:
            raise Exception("Wrong type")

    def getValue(self, state):
  
        state = from_numpy(state)
        with no_grad():
            value = self.critic_net(state)
        return value.item()

    def save(self, path):
       
        print(path)
        save(self.actor_net.state_dict(), path + '_actor.pkl')
        save(self.critic_net.state_dict(), path + '_critic.pkl')

    def load(self, path):
       
        actor_state_dict = load(path + '_actor.pkl')
        critic_state_dict = load(path + '_critic.pkl')
        self.actor_net.load_state_dict(actor_state_dict)
        self.critic_net.load_state_dict(critic_state_dict)

    def storeTransition(self, transition):
        
        self.buffer.append(transition)

    def trainStep(self, batchSize=None):
        if batchSize is None:
            if len(self.buffer) < self.batch_size:
                return
            batchSize = self.batch_size

        state = tensor([t.state for t in self.buffer], dtype=torch_float)
        action = tensor([t.action for t in self.buffer], dtype=torch_long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tensor([t.a_log_prob for t in self.buffer], dtype=torch_float).view(-1, 1)

        # Unroll rewards
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = tensor(Gt, dtype=torch_float)

        if self.use_cuda:
            state, action, old_action_log_prob = state.cuda(), action.cuda(), old_action_log_prob.cuda()
            Gt = Gt.cuda()

        for _ in range(self.ppo_update_iters):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batchSize, False):
                # Calculate the advantage at each step
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # Get the current prob
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                # PPO
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch_min(surr1, surr2).mean() #gradient descent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network use mse loss
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]


class Actor(nn.Module):
    def __init__(self, numberOfInputs, numberOfOutputs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(numberOfInputs, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.action_head = nn.Linear(256, numberOfOutputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, numberOfInputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(numberOfInputs, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.state_value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        value = self.state_value(x)
        return value
