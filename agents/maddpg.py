# main code that contains the neural network setup
# policy + critic updates
# see ddpg_agent.py for other details in the network

from agents.ddpg_agent import Agent
import torch
import numpy as np
from utils.utilities import soft_update, transpose_to_tensor, transpose_list
from buffers.ReplayBuffer import ReplayBuffer

device = 'cpu'

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 4        # Train agents every 4 steps 
NUM_UPDATES = 1         # How many times agents should be trained

class MADDPG:
    def __init__(self, state_size, action_size, seed):
        super(MADDPG, self).__init__()

        self.maddpg_agents = [Agent(state_size, action_size, seed),
                              Agent(state_size, action_size, seed)]
        
        self.t_step = 0

        self.memory = ReplayBuffer(2, BUFFER_SIZE, BATCH_SIZE, 0)
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.update_every = UPDATE_EVERY
        self.num_updates = NUM_UPDATES

    def save(self):
        for i in range(len(self.maddpg_agents)):
            torch.save(self.maddpg_agents[i].actor_local.state_dict(), 'models/checkpoint_actor_{}_final.pth'.format(i))
            torch.save(self.maddpg_agents[i].critic_local.state_dict(), 'models/checkpoint_critic_{}_final.pth'.format(i))

    def load(self):
        for i in range(len(self.maddpg_agents)):
            actor_file = 'models/checkpoint_actor_{}_final.pth'.format(i)
            critic_file = 'models/checkpoint_critic_{}_final.pth'.format(i)
            self.maddpg_agents[i].actor_local.load_state_dict(torch.load(actor_file))
            self.maddpg_agents[i].critic_local.load_state_dict(torch.load(critic_file))

    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()
            
    def act(self, all_states):
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.maddpg_agents, all_states)]
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        for s,a,r,ns,d in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s,a,r,ns,d)
            
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            for _ in range(self.num_updates):
                for agent in self.maddpg_agents:
                    experiences = self.memory.sample()
                    agent.learn(experiences, self.gamma)
