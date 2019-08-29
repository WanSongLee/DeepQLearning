import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from gym import wrappers
import numpy as np
import time
from utils.prioritized_memory import Memory as ReplayMemory
import datetime

from collections import namedtuple
import random

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("using gpu")
    device = torch.device('cuda')
else:
    print("using cpu")
    device = torch.device('cpu')


# In[ ]:


class DQN(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(observation_size, 512),
                                nn.ReLU(),
                                nn.Linear(512 , 256),
                                nn.ReLU(),
                                nn.Linear(256 , 128),
                                nn.ReLU(),
                                nn.Linear(128 , 64),
                                nn.ReLU(),
                                nn.Linear(64 , 32),
                                nn.ReLU(),
                                nn.Linear(32 , 16),
                                nn.ReLU(),
                                nn.Linear(16 , 8),
                                nn.ReLU(),
                                nn.Linear(8, action_size))

    def forward(self, observation):
        return self.fc(observation)
        
class Agent:
    def __init__(self):
        observation_size = 128
        action_size = 4
        
        self.q_network = DQN(observation_size, action_size).to(device)
        self.target_network = DQN(observation_size, action_size).to(device)
        
        # copy the weights from q_network to target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    # act greedily with respect to the Q-function
    def act(self, state):
        # state: 1d numpy arrray
        out = self.q_network(state)
        out = torch.max(out, 0)[1]
        return out


# In[ ]:


class Trainer:
    def __init__(self, env, agent, memory, learning_rate, batch_size, target_update_freq,
                 epsilon_start, epsilon_min, epsilon_decay_rate, epsilon_test):
        self.env = env
        self.agent = agent
        self.memory = memory
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_test = epsilon_test
        
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(agent.q_network.parameters(), learning_rate)
        
        # counter used to track when we need to update our target network
        self.update_counter = 0
        self.episode_counter = 0
        
        self.loss_history = []
        self.avg_loss_history = []
        self.frame_count = 0
    
    # add items to the prioritized replay memory
    def append_sample(self, state, action, next_state, reward, done):
        current_q = self.agent.q_network(state.to(device))[action]
        
        ###########single dqn#############
#         max_next_q = self.agent.target_network(next_state.to(device)).max()
#         expected_q = reward + (1-done) * (GAMMA * max_next_q)
        ###########single dqn#############

        ###########double dqn#############
        best_next_action = self.agent.q_network(next_state.to(device)).max(0)[1]
        best_next_q = self.agent.target_network(next_state.to(device))[best_next_action]
        expected_q = reward + (1-done) * (GAMMA * best_next_q)
        ###########double dqn#############
        
        # calculate and save the error to our memory, use for importance sampling
        error = abs((current_q - expected_q).item())
        self.memory.add(error, (state.numpy(), action, reward, next_state.numpy(), done))
    
    def learn(self):
        if self.memory.tree.n_entries < self.batch_size:
            return
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
        
        # get prioritized experience from memory 
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()
        
        states = mini_batch[0].tolist()
        actions = mini_batch[1].tolist()
        rewards = mini_batch[2].tolist()
        next_states = mini_batch[3].tolist()
        dones = mini_batch[4].tolist()
        
        # convert from numpy ndarray to pytorch tensor
        states = torch.from_numpy(np.stack(states)).to(device)
        actions = torch.IntTensor(actions).to(device)
        next_states = torch.from_numpy(np.stack(next_states)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q = self.agent.q_network(states).gather(1, actions.unsqueeze(1).long()).squeeze(1)
        
        ###########single dqn#############
#         max_next_q = self.agent.target_network(next_states).detach().max(1)[0]
#         expected_q = rewards + (1-dones) * (GAMMA * max_next_q)
        ###########single dqn#############
        
        ###########double dqn#############
        best_next_actions = self.agent.q_network(next_states).detach().max(1)[1]
        best_next_q = self.agent.target_network(next_states).gather(1, best_next_actions.unsqueeze(1)).squeeze(1).detach()
        expected_q = rewards + (1-dones) * (GAMMA * best_next_q)
        ###########double dqn#############
        
        # update error in memory
        errors = torch.abs(current_q - expected_q).cpu().data.numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
            
        # each transition has its own importance-sampling weight
        temp = F.smooth_l1_loss(current_q, expected_q, reduction ='none')
        loss = (torch.FloatTensor(is_weights).to(device) * temp).mean()
        
        self.loss_history.append(loss.item())
        avg_loss = self.loss_history[-1000:]
        self.avg_loss_history.append(sum(avg_loss)/len(avg_loss))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # complete an episode, for each transitions, add it to the replay memory, then learn from the memory
    def play_episode(self, test=False):
        self.episode_counter += 1
        state = self.env.reset()
        state = torch.from_numpy(state).float()

        if render:
            self.env.render()

        done = False
        score = 0
        while not done:
            if test:
                epsilon = self.epsilon_test
            else:
                epsilon = max(1 - self.epsilon_decay_rate * self.update_counter, self.epsilon_min)

            # select action using epsilon-greedy policy
            if random.random() < epsilon: # explore
                action = self.env.action_space.sample()
            else: # exploit
                action = self.agent.act(state.to(device)).item()
              
            # send our action to the environment
            next_state, reward, done, _ = self.env.step(action)
            next_state = torch.from_numpy(next_state).float()

            score += reward
            
            if render:
                self.env.render()
                
            # save our experience to the memory
            self.append_sample(state, action, next_state, reward, done)
            
            if not test:
                self.learn()

            state = next_state
        
        return score


# In[ ]:


def save_checkpoint(PATH, trainer, train_scores, avg_train_scores,
                    test_scores, avg_test_scores, total_time):
    agent = trainer.agent
    memory = trainer.memory
    sumtree = memory.tree
    
    torch.save({
                'memory_e': memory.e,
                'memory_a': memory.a,
                'memory_beta': memory.beta,
                'memory_capacity': memory.capacity,
                'sumtree_tree': sumtree.tree,
                'sumtree_data': sumtree.data,
                'sumtree_nentries': sumtree.n_entries,
                
                'agent_qnetwork': agent.q_network.state_dict(),
                'agent_targetnetwork': agent.target_network.state_dict(),
        
                'trainer_learnrate': trainer.learning_rate,
                'trainer_batchsize': trainer.batch_size,
                'trainer_updatefreq': trainer.target_update_freq,
                'trainer_epsilonstart': trainer.epsilon_start,
                'trainer_epsilonmin': trainer.epsilon_min,
                'trainer_epsilon_decay_rate': trainer.epsilon_decay_rate,
                'trainer_epsilon_test': trainer.epsilon_test,
                
                'trainer_optimizer': trainer.optimizer.state_dict(),
                'trainer_losshistory': trainer.loss_history,
                'trainer_avglosshistory': trainer.avg_loss_history,
                'trainer_episodecounter': trainer.episode_counter,
                'trainer_updatecounter': trainer.update_counter,
                
                'trainscores': train_scores,
                'avgtrainscores': avg_train_scores,
                'testscores': test_scores,
                'avgtestscores': avg_test_scores,
                'totaltime': total_time
                }, PATH)

def load_checkpoint(PATH, env):
    checkpoint = torch.load(PATH)
    
    memory = ReplayMemory(checkpoint['memory_capacity'])
    memory.e = checkpoint['memory_e']
    memory.a = checkpoint['memory_a']
    memory.beta = checkpoint['memory_beta']
    memory.tree.tree = checkpoint['sumtree_tree']
    memory.tree.data = checkpoint['sumtree_data']
    memory.tree.n_entries = checkpoint['sumtree_nentries']

    agent = Agent()
    agent.q_network.load_state_dict(checkpoint['agent_qnetwork'])
    agent.target_network.load_state_dict(checkpoint['agent_targetnetwork'])
    
    trainer = Trainer(env, agent, memory, 
                     checkpoint['trainer_learnrate'], 
                     checkpoint['trainer_batchsize'], 
                     checkpoint['trainer_updatefreq'], 
                     checkpoint['trainer_epsilonstart'], 
                     checkpoint['trainer_epsilonmin'], 
                     checkpoint['trainer_epsilon_decay_rate'],
                     checkpoint['trainer_epsilon_test'])
    trainer.optimizer.load_state_dict(checkpoint['trainer_optimizer'])
    trainer.loss_history = checkpoint['trainer_losshistory']
    trainer.avg_loss_history = checkpoint['trainer_avglosshistory']
    trainer.episode_counter = checkpoint['trainer_episodecounter']
    trainer.update_counter = checkpoint['trainer_updatecounter']
        
    train_scores = checkpoint['trainscores']
    avg_train_scores = checkpoint['avgtrainscores'] 
    test_scores = checkpoint['testscores']
    avg_test_scores = checkpoint['avgtestscores']
    total_time = checkpoint['totaltime']
    
    print("learning rate", trainer.learning_rate)
    print("batch_size", trainer.batch_size)
    print("target_update_freq", trainer.target_update_freq)
    print("epsilon_start", trainer.epsilon_start)
    print("epsilon_min", trainer.epsilon_min)
    print("epsilon_decay_rate", trainer.epsilon_decay_rate)
    print("memory capacity", memory.capacity)
    
    print(agent.q_network)
    print("number of parameters:" ,sum([p.numel() for p in agent.q_network.parameters()]))
    
    return trainer, train_scores, avg_train_scores, test_scores, avg_test_scores, total_time 


# In[ ]:


env = gym.make('Breakout-ram-v0')
record = True
if record:
    struct = time.localtime(time.time())
    video_path = './videos/BreakoutRam/'
    video_path += "{0}-{1} {2}-{3}-{4}".format(struct.tm_mon,struct.tm_mday,struct.tm_hour,struct.tm_min,struct.tm_sec)
    print('saving recorded videos to path "{0}"'.format(video_path))
    env = wrappers.Monitor(env, video_path, video_callable=lambda episode_id: True, force=True)

PATH = 'checkpoint/BreakoutRam.checkpoint'
load = False

if load:
    trainer, train_scores, avg_train_scores, test_scores, avg_test_scores, total_time = load_checkpoint(PATH, env)
    env.episode_id = trainer.episode_counter
else: 
    train_scores, avg_train_scores, test_scores, avg_test_scores = [], [], [], []
    total_time = 0
    
    learning_rate = 0.00001
    batch_size = 32
    target_update_freq = 10000
    epsilon_start = 1
    epsilon_min = 0.1
    epsilon_decay_rate = 0.0000009
    epsilon_test = 0.0005

    memory = ReplayMemory(1000000)
    agent = Agent()
    trainer = Trainer(env, agent, memory, learning_rate, batch_size, target_update_freq,
                     epsilon_start, epsilon_min, epsilon_decay_rate, epsilon_test)
    print(agent.q_network)
    print("number of parameters:" ,sum([p.numel() for p in agent.q_network.parameters()]))


# In[ ]:


GAMMA = 0.99
render = False
    
test_every = 20
save_every = 200

fig = plt.figure()
train_scores_ax = fig.add_subplot(311)
test_scores_ax = fig.add_subplot(312)
loss_ax = fig.add_subplot(313)
plt.ion()

previous_time = time.time()
for i in range(100000):
    if trainer.episode_counter % test_every == 0:
        score = trainer.play_episode(test=True)
        test_scores.append(score)
        
        avg_score = test_scores[-100:]
        avg_test_scores.append(sum(avg_score)/len(avg_score))
        
        print("Episode {0} finished with score of {1}"
                  .format(trainer.episode_counter, score))
    else:
        score = trainer.play_episode(test=False)
        train_scores.append(score)

        avg_score = train_scores[-100:]
        avg_train_scores.append(sum(avg_score)/len(avg_score))
        print("Episode {0}({1}) finished with score of {2}, after learning from {3} frames"
                  .format(trainer.episode_counter, trainer.episode_counter - len(test_scores), score, trainer.update_counter))
    
    if trainer.episode_counter % save_every == 0 and i > 0:
        current_time = time.time()
        total_time += current_time - previous_time
        save_checkpoint(PATH, trainer, train_scores, avg_train_scores,
                        test_scores, avg_test_scores, total_time)
        print("took {0} seconds to run the last 200 episodes".format(current_time - previous_time, save_every))
        print("took {0} to run everything".format(datetime.timedelta(seconds = int(total_time))))
        previous_time = current_time
        
    
trainer.env.close()


# In[ ]:









