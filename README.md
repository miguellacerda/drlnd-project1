## Udacity Deep Reinforcement Learning Nanodegree 
## Project 1: Navigation

### Description of Environment

In this project, an agent is trained to navigate a square world to collect as many yellow bananas as possible while avoiding blue bananas. To this effect, the agent receives a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. The action set contains four discrete actions, namely "move forward" (0), "move backward" (1), "turn left" (2) and "turn right" (3). 

The task is episodic where each episode terminates after 300 steps. The environment is considered solved when the trained agent obtains an average score of 13 over 100 consecutive episodes.


### Installation Instructions and Dependencies

The code is written in PyTorch and Python 3.6. I trained the agent on a MacBook Pro with a 2.9 GHz Intel Core i7 processor and 16GB of RAM. I did not use the GPU for training as it is not CUDA compatable. 

Follow the instructions below to run the code in this respository:

1. Create and activate a new environment with Python 3.6
    
   ###### Linux or Mac:
   
    `conda create --name drlnd python=3.6`
    
    `source activate drlnd`

   ###### Windows:

    `conda create --name drlnd python=3.6`
    
    `activate drlnd`

1. Install of OpenAI gym in the environment

   `pip install gym`
 
1. Install the classic control and box2d environment groups

   `pip install 'gym[classic_control]'`
   
   `pip install 'gym[box2d]'`

1. Clone the following repository and install the additional dependencies

   `git clone https://github.com/udacity/deep-reinforcement-learning.git`
   
   `cd deep-reinforcement-learning/python`
   
   `pip install .`

1. Download the Unity environment (available [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) for macOS)


### Code Description and Final Weights

All of the files below can be found in the code folder.

##### Module Descriptions

- `model.py` defines the Q-network architecture
- `dqn_agent.py` defines a DQN agent
- `double_dqn_agent.py` defines a double DQN agent
- `prioritised_ double_dqn_agent.py` defines a double DQN agent with prioritised experience replay

##### Training the Agent

Both of these files can be run by invoking python 3 at the command line:

- `project1_doubleDQN.py` is used to train a DQN or double DQN agent. On line 2, you can import the agent from either the `dqn_agent` or `double_dqn_agent` modules mentioned above
- `project1_prioritisedDoubleDQN.py` is used to train a double DQN agent with prioritised experience replay


##### The Trained Agent

- `trained_agent_weights.pth` contains the weights of the best Q-network (see Report.md)
- `trained_agent.py` loads the optimised weights and runs the trained agent for 1 episode

   
