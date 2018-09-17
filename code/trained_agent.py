from unityagents import UnityEnvironment
from double_dqn_agent import Agent
import torch

### ENVIRONMENT ###
env = UnityEnvironment(file_name="Banana.app", no_graphics=False)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state_size = brain.vector_observation_space_size

### AGENT ####
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

### load weights ####
weight_dict = torch.load('trained_agent_weights.pth')
agent.qnetwork_local.load_state_dict(weight_dict)
agent.qnetwork_target.load_state_dict(weight_dict)


state = env_info.vector_observations[0]                     # get the current state
score = 0
while True:
    action = agent.act(state, eps=0.0)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]            # get the next state
    reward = env_info.rewards[0]                            # get the reward
    done = env_info.local_done[0]                           # see if episode has finished
    agent.step(state, action, reward, next_state, done)     # update local Q-net, and soft update target Q-net
    score += reward                                         # update the score                                          
    state = next_state                                      # roll over the state to next time step
    if done:                                                # exit loop if episode finished
        break

print('Score =', score)
env.close()

