# Deep Reinforcement Learning Agent for Tennis App 

## Implementation Details 

Implemented and trained MADDPG algorithm for solving Reacher environment. Here's the gif of trained agent

![MADDPG Trained Agent](resources/trained_agent.gif)

### DDPG 

The implementation of dqn is in ```agents/maddpg.py``` and the trained model fro both the agents can be found at
* ```models/checkpoint_actor_0_final.pth```
* ```models/checkpoint_actor_1_final.pth```
* ```models/checkpoint_critic_0_final.pth```
* ```models/checkpoint_critic_1_final.pth```

It is inspired by the original paper and here's the snapshot of the algorithm from the paper:

![DDPG Algorithm](resources/maddpg.png)

#### Actor Architecture
```
Actor(
  (fc1): Linear(in_features=24, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=2, bias=True)
  (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

#### Critic Architecture
```
Critic(
  (fcs1): Linear(in_features=24, out_features=512, bias=True)
  (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=514, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=1, bias=True)
)
```

These are other the key pieces of the algorithm:

### Experience Replay 

Experience replay allows the RL agent to learn from past experiences. Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains experience tuples with the state, action, reward, and next state ```(s, a, r, s')```. The agent randomly samples from this buffer as part of the training. Random samplaing helps with the problem of correlated data. This prevents action values from oscillating, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found here in the ```buffers/ReplayBuffer.py``` file of the source code.

### Target Network 

Iterative update that adjusts the action-values towards target values that are only periodically updated, thereby reducing correlations with the target.

The target values are updated based on this equation. 
```
 θ_target = τ*θ_local + (1 - τ)*θ_target
```

You can find logic implemented in ```soft_update()``` method in ```dqn_agent.py``` of the source code. 

## Hyperparameters 

The agent uses these parameters
```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 4        # Train agents every 4 steps 
NUM_UPDATES = 1         # How many times agents should be trained
TAU = 0.02              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

The training part uses these paramters
```
Number of training episodes = 3000
Max number of steps in an episode = 1000
```
## Results

I was able to achieve the score of 0.5 in 938 episodes

<img src="resources/maddpg_score.png">

### Observations 
* Batch Normalization and slowing the learning rate helped with faster training 

## Ideas for future work 
- Tuning of hyperparameters for the network 
- Use Prioritized Experience Replay ([Link](https://arxiv.org/pdf/1511.05952.pdf))
- Implement Multi Agent version of PPO
