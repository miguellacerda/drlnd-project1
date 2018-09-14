## Udacity Deep Reinforcement Learning Nanodegree 
## Project 1: Navigation

### Description of Implementation

The environment is solved using a deep Q-network with fixed targets and experience replay. Training proceeds as follows:

1. The agent receives a state vector from the enviroment
1. Based on the current state, the agent choose an action that is epsilon-greedy with respect to the target Q-network
1. The agent then receives the next state vector and a reward from the environment (as well as a termination signal that indicates if the episode is complete)
1. The experience tuple `(state, action, reward, next state)` is added to the replay buffer
1. Every fixed number of steps, a sample is drawn from the replay buffer (assuming it contains enough tuples)
1. The sample is used to update the weights of the local Q-network, which are in turn used to adjust the weights of the target Q-network using a soft-update rule
1. The state that was observed in step (3) then becomes the current state and the processes repeats from step (2)

#### Learning Algorithms

This project considers three variations of the deep Q-learning algorithm:

1. Vanilla DQN with fixed targets and experience replay
1. Double DQN with fixed targets and experience replay
1. Double DQN with fixed targets and prioritised experience replay

#### Agent Hyperparameters

##### All Deep Q-Networks

- `epsilon` controls the degree of exploration vs exploitation of the agent in selecting its actions. `epsilon = 0` implies that the agent is greedy with respect to the Q-network (pure exploitation) and `epsilon = 1` implies that the agent selects actions completely randomly (pure exploration). In this project, `epsilon` is annealed from 1.0 to 0.1 in steps of 0.001 after each episode, and remains fixed at 0.1 therafter.
- `GAMMA = 0.99` is the discount factor that controls how far-sighted the agent is with respect to rewards. `GAMMA = 0` implies that only the immediate reward is important and `GAMMA = 1.0` implies that all rewards are equally important, irrespective whether they are realised soon and much later
- `TAU = 0.001` controls the degree to which the target Q-network parameters are adjusted toward those of the local Q-network. `TAU = 0` implies no adjustment (the target Q-network does not ever learn) and `TAU = 1` implies that the target Q-network parameters are completelty replaced with the local Q-network parameters
- `LR = 0.0005` is the learning rate for the gradient descent update of the local Q-network weights
- `UPDATE_EVERY = 4` determines the number of sampling steps between rounds of learning (Q-network parameter updates)
- `BUFFER_SIZE = 10000` is the number of experience tuples `(state, action, reward, next_state, done)` that are stored in the replay buffer and avaiable for learning
- `BATCH_SIZE = 64` is the number of tuples that are sampled from the replay buffer for learning

##### Prioritised Experience Replay

- `e_priority = 0.01` is added to the absolute value of the TD error to ensure that none of the priorities are exactly zero. This ensures that all tuples in the replay buffer have a non-zero probability of being selected for training
- `a_priority` controls the extent to which the TD error influences the probability of selecting a tuple for training. `a_priority=0` implies that all tuples in the buffer have equal probability of selection, while `a_priority=1` implies pure priority (TD error-based) sampling. We set `a_priority = 0.6` as in [this paper](https://arxiv.org/pdf/1511.05952.pdf)
- `b_priority` controls the extent to which the biased sampling from the replay buffer is corrected in the gradient descent update. `b_priority = 0` implies no correction for bias and `b_priority = 1` implies complete bias correction. In this project, `b_priority` is increased from 0.4 to 1.0 in steps of 0.0005 after each episode, and remains fixed at 1.0 therafter (as done in [this paper](https://arxiv.org/pdf/1511.05952.pdf))


#### Model Architecture and Hyperparameters

### Results

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment. 

### Future Plans for Improvement

The submission has concrete future ideas for improving the agent's performance.

- duelling
- hyperpameter optimisation
- code optimation - especialy prioritsed replay
