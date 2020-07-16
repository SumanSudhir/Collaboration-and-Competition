import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from actor_critic import Actor,Critic
class Agent():
	"""Interacts with and learns from the environment."""
	
	def __init__(self, state_size=24, action_size=2, BATCH_SIZE=128, BUFFER_SIZE = int(1e6), discount_factor=1, tau=1e-2, noise_coefficient_start=5, noise_coefficient_decay=0.99, LR_ACTOR=1e-3, LR_CRITIC=1e-3, WEIGHT_DECAY=1e-3, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
		"""
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			BATCH_SIZE (int): mini batch size
			BUFFER_SIZE (int): experience storing lenght, keep it as high as possible
			discount_factor (float): discount factor for calculating Q_target
			tau (float): interpolation parameter for updating target network
			noise_coefficient_start (float): value to be multiplied to OUNoise sample
			noise_coefficient_decay (float): exponential decay factor for value to be multiplied to OUNoise sample
			LR_ACTOR (float): learning rate for actor network
			LR_CRITIC (float): learning rate for critic network
			WEIGHT_DECAY (float): Weight decay for critic network optimizer
			device : "cuda:0" if torch.cuda.is_available() else "cpu"
		"""
        
		self.state_size = state_size
		print(device)
		self.action_size = action_size
		self.BATCH_SIZE = BATCH_SIZE
		self.BUFFER_SIZE = BUFFER_SIZE
		self.discount_factor = discount_factor
		self.tau = tau
		self.noise_coefficient = noise_coefficient_start
		self.noise_coefficient_decay = noise_coefficient_decay
		self.steps_completed = 0
		self.device = device
		# Actor Network (w/ Target Network)
		self.actor_local = Actor(state_size, action_size).to(self.device)
		self.actor_target = Actor(state_size, action_size).to(self.device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

		# Critic Network (w/ Target Network)
		self.critic_local = Critic(state_size, action_size).to(self.device)
		self.critic_target = Critic(state_size, action_size).to(self.device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
				
		# Noise process
		self.noise = OUNoise((1, action_size))

		# Replay memory
		self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE)
	
	def step(self, state, action, reward, next_state, done, agent_number):
		"""Save experience in replay memory, and use random sample from buffer to learn."""
		self.memory.add(state, action, reward, next_state, done)
		self.steps_completed += 1
		# If number of memory data > Batch_Size then learn
		if len(self.memory) > self.BATCH_SIZE:
			experiences = self.memory.sample(self.device)
			self.learn(experiences, self.discount_factor, agent_number)

	def act(self, states, add_noise):
		"""Returns actions for given state as per current policy."""
		states = torch.from_numpy(states).float().to(self.device)
		actions = np.zeros((1, self.action_size))   # shape will be (1,2)
		self.actor_local.eval()
		with torch.no_grad():
			actions[0, :] = self.actor_local(states).cpu().data.numpy()
		self.actor_local.train()
		if add_noise:
			actions += self.noise_coefficient * self.noise.sample()
		return np.clip(actions, -1, 1)

	def reset(self):
		self.noise.reset()

	def learn(self, experiences, discount_factor, agent_number):
		"""Update policy and value parameters using given batch of experience tuples.
		Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
		where:
			actor_target(state) -> action
			critic_target(state, action) -> Q-value
		Params
		======
			experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
			discount_factor (float): discount factor
		"""
		states, actions, rewards, next_states, dones = experiences

		# ---------------------------- update critic ---------------------------- #
		# Get predicted next-state actions and Q values from target models
		actions_next = self.actor_target(next_states)
		
		# It is basically taking action of both the agents, so if agent_number=0 then we will have to concatenate agent0 action(currently actions_next) and agent1 action(currently actions[:,2:])
		if agent_number == 0:    
			actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)   
		else:
			actions_next = torch.cat((actions[:,:2], actions_next), dim=1)

		Q_targets_next = self.critic_target(next_states, actions_next)
		# Compute Q targets for current states (y_i)
		Q_targets = rewards + (discount_factor * Q_targets_next * (1 - dones))
		# Compute critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# ---------------------------- update actor ---------------------------- #
		# Compute actor loss
		actions_pred = self.actor_local(states)
		
		if agent_number == 0:
			actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
		else:
			actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)

		actor_loss = -self.critic_local(states, actions_pred).mean()
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# ----------------------- update target networks ----------------------- #
		self.soft_update(self.critic_local, self.critic_target)
		self.soft_update(self.actor_local, self.actor_target)                     

		# Update noise_coefficient value
		# self.noise_coefficient = self.noise_coefficient*self.noise_coefficient_decay

		self.noise_coefficient = max(self.noise_coefficient-(1/self.noise_coefficient_decay),0)
		# print(self.steps_completed,': ',self.noise_coefficient)
		
	def soft_update(self, local_model, target_model):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
			local_model: PyTorch model (weights will be copied from)
			target_model: PyTorch model (weights will be copied to)
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
			
			
class OUNoise:
	"""Ornstein-Uhlenbeck process."""

	def __init__(self, size, mu=0.0, theta=0.13, sigma=0.2):
		"""Initialize parameters and noise process."""
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.size = size
		self.reset()

	def reset(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)

	def sample(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
		self.state = x + dx
		return self.state
	
class ReplayBuffer:
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, action_size, buffer_size, batch_size): 
		"""    
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
		"""
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
	
	def add(self, state, action, reward, next_state, done):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)
	
	def sample(self,device):
		"""Randomly sample a batch of experiences from memory."""
		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

		return (states, actions, rewards, next_states, dones)
	
	def __len__(self):
		return len(self.memory)