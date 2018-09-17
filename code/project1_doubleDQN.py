from unityagents import UnityEnvironment
from double_dqn_agent import Agent
#from dqn_agent import Agent # to use a DQN agent without double DQN

import torch
import numpy as np

### ENVIRONMENT ###
env = UnityEnvironment(file_name="Banana.app", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

### AGENT ####
agent = Agent(state_size=state_size, action_size=action_size, seed=0)


#### TRAINING AGENT ####
num_episodes = 2000
epsilon = 1.0
epsilon_decay = 0.995 #1/1000
epsilon_min = 0.01 #0.1

scores = []
last_mean_score = 13

for episode in range(num_episodes):
    
    env_info = env.reset(train_mode=True)[brain_name]           # reset the environment
    state = env_info.vector_observations[0]                     # get the current state
    score = 0                                                   # initialize the score

    while True:
        action = agent.act(state, epsilon)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]            # get the next state
        reward = env_info.rewards[0]                            # get the reward
        done = env_info.local_done[0]                           # see if episode has finished
        agent.step(state, action, reward, next_state, done)     # update local Q-net, and soft update target Q-net
        score += reward                                         # update the score                                          
        state = next_state                                      # roll over the state to next time step
        if done:                                                # exit loop if episode finished
            scores.append(score)
            break

    epsilon = max(epsilon_decay**episode, epsilon_min)  #max(epsilon-epsilon_decay, epsilon_min)   # adjust epsilon
    print('Episode:', episode, 'Epsilon:', epsilon, '- Score:', score)
    
    if len(scores) >= 100:
        mean_score = np.mean(scores[-100:])
        if mean_score > last_mean_score:
            torch.save(agent.qnetwork_local.state_dict(), 'exponentialEpsilon_doubleDQN_64x64_'+str(episode)+'.pth')
            last_mean_score = mean_score

    
env.close()






















