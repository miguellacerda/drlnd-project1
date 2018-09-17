from unityagents import UnityEnvironment
from prioritised_double_dqn_agent import Agent
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


#### TRAINING THE AGENT ####
num_episodes = 2000

# epsilon controls exploration/exploitation trade-off: 
# epsilon=0 implies pure exploitation (best action always), epsilon = 0 implies pure exploration (completely random actions)
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# b_priority corrects the gradient descent update step for biased sampling
# power of the weights: b_priority = 0 implies no corrections, b_priority=1 implies full correction
b_priority = 0.4
b_step = 1/2000 
b_max = 1.0

scores = []
last_mean_score = 13 # save weights if score greater than this

for episode in range(num_episodes):
    
    env_info = env.reset(train_mode=True)[brain_name]                                   # reset the environment
    state = env_info.vector_observations[0]                                             # get the current state
    score = 0                                                                           # initialize the score

    while True:
        action = agent.act(state, epsilon)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]                                    # get the next state
        reward = env_info.rewards[0]                                                    # get the reward
        done = env_info.local_done[0]                                                   # see if episode has finished
        priority = agent.priority(state, action, reward, next_state)                    # compute priority for this experience tuple
        agent.step(state, action, reward, next_state, done, priority, b_priority)       # update local Q-net, and soft update target Q-net
        score += reward                                                                 # update the score                                          
        state = next_state                                                              # roll over the state to next time step
        if done:                                                                        # exit loop if episode finished
            scores.append(score)
            break
    
    epsilon = max(epsilon_decay**episode, epsilon_min) #max(epsilon-epsilon_step, epsilon_min)    # adjust epsilon
    b_priority = min(b_priority+b_step, b_max)          # adjust b_priority
    print('Episode:', episode, '- Score:', score)
    
    if len(scores) >= 100:
        mean_score = np.mean(scores[-100:])
        if mean_score > last_mean_score:
            torch.save(agent.qnetwork_local.state_dict(), 'exponentialEpsilon_prioritised_double_DQN_64x64_'+str(episode)+'.pth')
            last_mean_score = mean_score
          
    
env.close()


