import numpy as np
import random
from collections import namedtuple, deque
''' deque = double-ended queue; can add and remove from either end 
    namedtuple = experience instance that is added to replay buffer '''

from model import QNetwork

import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
E_PRIORITY = 0.01        # priority offset to prevent zero probability of selection
A_PRIORITY = 0.6        # affects degree of sampling bias: a = 0 implies uniform sampling, a = 1 implies pure priority sampling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def WeightedMSE(input, target, weights=1):
    return torch.sum(weights*(input-target)**2) # prioritised replay


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def priority(self, states, actions, rewards, next_states):
        if len(states.shape) == 1: 
            # only a single experience tuple to evaluate
            # need to format variables accordingly:
            states = torch.from_numpy(states).float().unsqueeze(0).to(device)
            next_states = torch.from_numpy(next_states).float().unsqueeze(0).to(device)
            rewards = torch.tensor([[rewards]], dtype=torch.float).to(device) # scalar value
            actions = torch.tensor([[actions]], dtype=torch.uint8).to(device) # scalar value
        
        action_local = self.qnetwork_local.forward(next_states).argmax(1)
        max_q = self.qnetwork_target.forward(next_states)[np.arange(action_local.shape[0]), action_local]    
        delta = (rewards.squeeze() + GAMMA*max_q) - self.qnetwork_local(states)[np.arange(actions.shape[0]),actions.byte().squeeze().cpu().numpy()] 
        priority = torch.abs(delta) + E_PRIORITY
        return priority.squeeze().tolist()

        
    def step(self, state, action, reward, next_state, done, priority, b_priority):
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, priority)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(b_priority) # needs b_priority to compute weights
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, experience_indices = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"        
        actions_local = self.qnetwork_local.forward(next_states).argmax(1)
        max_q_values = self.qnetwork_target.forward(next_states)[np.arange(actions_local.shape[0]), actions_local]
        td_target = rewards.squeeze()+gamma*max_q_values*(1-dones.squeeze())
        
        predicted_q_values = self.qnetwork_local.forward(states)
        predicted_q_values = predicted_q_values[np.arange(predicted_q_values.shape[0]),actions.squeeze()]
        
        self.optimizer.zero_grad() # must zero the gradients each time, otherwise they get summed
        
        # Forward and backward passes
        loss = WeightedMSE(predicted_q_values, td_target, weights)
        loss.backward() # backward pass to compute the gradients
        self.optimizer.step() # take a step using the learning rate and computed gradient
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # ------------------- update priorities in the replay buffer ------------------- #
        new_priorities = self.priority(states, actions, rewards, next_states)        
        for count, idx in enumerate(experience_indices):
            self.memory.memory[idx] = self.memory.memory[idx]._replace(priority=new_priorities[count])
                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)
    
    def sample(self, b_priority):
        """Prioritised sampling of a batch of experiences from memory."""

        # sampling probabilities:        
        p = np.array([self.memory[i].priority for i in range(len(self.memory))])**A_PRIORITY
        p /= p.sum()

        # sample experiences based on priority probabilities:
        experience_indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=p)
        experiences = [self.memory[i] for i in experience_indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy((len(self.memory)*p[experience_indices])**(-b_priority)).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, experience_indices)
        # also return experience indices so that we can update their priorities

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)