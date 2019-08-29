# DeepQLearning
Let’s create an AI that learns to play Atari Breakout using RAM inputs and Deep Q-Networks.

## Table of contents:
1. Reinforcement Learning
    1. RL vs Supervised vs Unsupervised
    2. Terminology
    3. Markov Decision Process
    4. Return
    5. Action Value Function
    6. Bellman Equation
    7. Q-Learning Algorithm
    8. Exploration-Exploitation Tradeoff
    9. Q-function Approximation using Neural Networks
2. Results
    1. Model Architecture and Hyperparameter
3. Optimizations
    1. Huber Loss
    2. Experience Replay
    3. Prioritized Experience Replay
    4. Fixed Target Networks
    5. Double DQN
    
## Reinforcement Learning
Reinforcement learning is an area of Machine Learning concerned with how agents should take actions in an environment to maximize some notion of cumulative reward.

### RL vs Supervised vs Unsupervised
Reinforcement learning is a different paradigm in machine learning. In a supervised learning setting, we have some training data, use the data to train our model, then we can use it to predict new unlabelled data. There is always a correct answer for classification problems and a target for regression problems. Examples of supervised learning are predicting housing prices and image classification. In an unsupervised learning setting, we have a lot of data, and we can use it to find hidden patterns in the data. Examples of unsupervised learning include word2vec and clustering. For reinforcement learning, we are just trying to maximize reward in an environment, it does not fit into supervised or unsupervised settings.

> **_SIDENOTE: Can we train our agent using supervised learning?_** 
    </br>
    Can we train our agent by showing it examples of an expert interacting with the environment and ask our agent to imitate it?
    </br>
    The main problem is that our agent will only learn strategies that our expert uses, our agent will not be able to learn more optimal strategies. The second problem is that there are many possible paths an agent could take. Each path could be equally valid, so there is not necessarily a right/wrong action at each timestep. 
    </br>
    Supervised learning alone will not give us a good result, but it can be combined with reinforcement learning to help train the agent. The AlphaStar team used supervised learning to bootstrap the training of their StarCraft agent and then used reinforcement learning to improve the performance <sup> <a name="firstb"> [\[1\]](#firsta)</a> </sup>
    
    
### Terminology

In Reinforcement learning, we have an **agent** that interacts with the **environment**.

At every interaction/timestep:
1. The agent receives the **state** <img src="https://latex.codecogs.com/svg.latex?s_t" /> from the environment
2. The agent will use the state to make an **action** <img src="https://latex.codecogs.com/svg.latex?a_t" /> based on the state, then sends it to the environment
3. The environment will update its internal state, then returns a **reward** <img src="https://latex.codecogs.com/svg.latex?r_t" />, which depends on the state and action

This loop repeat until the **terminal state** (in Breakout, this would be when the player loses all their lives or wins the game by breaking all of the bricks)

The goal of reinforcement learning is to learn a "good" **policy** <img src="https://latex.codecogs.com/svg.latex?\pi_\theta(a_t|s_t)" /> (we will explain what good means). The policy tells us what action to take when we are in a certain state.

The **trajectory**(**episode**, **rollout**) <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}" /> is a sequence of states and actions in the world 

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?(s_0,%20a_0,%20s_1,%20a_1,%20s_2,%20a_2,...)" /> </p>

### Markov Decision Process

We will represent our environment as a **Markov Decision Process**. This means that our representation of the state needs to satisfy the **Markov assumption**. The Markov assumption states that all relevant information is encapsulated in the current state, i.e. the future is independent of the past given the present.

Markov assumption is a little confusing, so let us go through an example. Let's say that we are playing a game of Breakout and we decide to use the current image frame to represent our state. The frame tells us where the bricks/ball/player is, but our action should also depend on the ball's velocity. If the ball is moving towards bottom-left, we should move left as well. To encapsulate the the ball's velocity, we can represent our state as a stack of last few frames. Thus the Markov assumption would be satisfied.

For this project, we are not dealing with image inputs, we are dealing with RAM inputs. The RAM inputs are a 128-byte array that is used by the emulator to represent the state of the game. We can guess that some of the bytes will be used to store which bricks are still available, numbers of lives, the position and velocity of the player and ball, etc. Because the RAM state is also used by the emulator to represent the state of the game, it is guaranteed to satisfy the Markov assumption as all relevant information is encapsulated in the RAM state.

### Return
We want a policy a that will give us "good" trajectory on average, but how do we differentiate good trajectories from bad ones? The simplest way to evaluate a trajectory is to sum up the total reward. This is called **finite-horizon undiscounted return**.

<p align="center"> <img src="http://latex.codecogs.com/svg.latex?R(\mathcal{T})%20=%20\sum_{t=0}^Tr_t" /> </p>

There is a slight problem with this return. If there is a cycle (an agent does a sequence of actions that causes to the environment to return back to the same state), the return can become infinite with no way for us to compare two different trajectories. For example, there there is an agent that always get a reward of +1 in an environment, and another agent always get a reward of +2. Obviously, the second agent's policy should be encouraged, but we cannot compare their return since they both diverge to infinity.

We can solve this problem using **infinite-horizon discounted return**.

<p align="center"> <img src="http://latex.codecogs.com/svg.latex?R(\mathcal{T})%20=%20\sum_{t=0}^{\infty}\gamma^tr_t" /> </p>

The only difference between the discounted return and undiscounted return is that there is a power of **discount factor** <img src="http://latex.codecogs.com/svg.latex?\gamma^t" /> multiplied with each reward. The discount factor is a value between 0 and 1, it determines how important the future is compared to the present. This means that an immediate reward is better than a future reward. When <img src="http://latex.codecogs.com/svg.latex?\gamma" /> is close to 0, the agent is myopic and only cares about the immediate reward. When <img src="http://latex.codecogs.com/svg.latex?\gamma" /> is close to 1, the agent is farsighted and cares about all of the future rewards equally.

The structure of the discounted return has a nice mathematical property, we can recursively breakdown a return into the immediate reward <img src="http://latex.codecogs.com/svg.latex?r_t" /> + discounted future return <img src="http://latex.codecogs.com/svg.latex?\gamma%20 R(\mathcal{T'})" />. We will exploit this property soon.

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\begin{align*}%20R(\mathcal{T})&=\sum_{t=0}^\infty%20\gamma^tr_t\\%20&=%20r_t+\gamma%20r_{t+1}+\gamma^2r_{t+2}+\gamma^3r_{t+3}+...\\%20&=%20r_t+\gamma%20(r_{t+1}+\gamma%20r_{t+2}+\gamma^2r_{t+3}+...)\\%20&=%20r_t+\gamma%20R(\mathcal{T}%27)\\%20\end{align*}" />
</p>

### Action-Value Function

**On-policy action value function** is the expected return if we start in state <img src="https://latex.codecogs.com/svg.latex?s_t" />, take any arbitrary action <img src="https://latex.codecogs.com/svg.latex?a" />(does to have to come from our policy), then always act according to our policy <img src="https://latex.codecogs.com/svg.latex?\pi" /> afterwards.

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?Q^{\pi}(s,a)=\underset{\tau%20\sim%20\pi}{\mathrm{E}}\left[R(\tau)%20|%20s_{0}=s,%20a_{0}=a\right]"/>
</p>

**Optimal action value function** <img src="http://latex.codecogs.com/svg.latex?Q^*(s,a)"/> is the expected return if we start in state <img src="https://latex.codecogs.com/svg.latex?s_t" />, take any arbitrary action a, then always act according to the optimal policy in the environment afterwards.

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?Q^{*}(s,%20a)=\max%20_{\pi}%20\underset{\tau%20\sim%20\pi}{\mathrm{E}}\left[R(\tau)%20|%20s_{0}=s,%20a_{0}=a\right]"/>
</p>

The action value function is also called <img src="http://latex.codecogs.com/svg.latex?Q" />-function, it is the function that our algorithm is trying to find. Once we have the optimal <img src="http://latex.codecogs.com/svg.latex?Q" />-function, we can always act perfectly to maximize our reward. At every timestep, we select the action that gives us the highest <img src="http://latex.codecogs.com/svg.latex?Q" />-value of for our current state. If we repeat this process until the terminal state, we will get the perfect trajectory possible.

But how can we find this optimal <img src="http://latex.codecogs.com/svg.latex?Q" />-function? We cannot possibly try every policy as the definition suggest. We need a different but equivalent definition for this <img src="http://latex.codecogs.com/svg.latex?Q" />-function.

### Bellman Equation

Remember the recursive definition for our return? This is used in the **optimal Bellman equation** to help us find <img src="http://latex.codecogs.com/svg.latex?Q^*" />.

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?Q^{*}(s,%20a)=\underset{s^{\prime}%20\sim%20P}{\mathrm{E}}\left[r(s,%20a)+\gamma%20\max%20_{a^{\prime}}%20Q^{*}\left(s^{\prime},%20a^{\prime}\right)\right]" />
</p>

This equation looks intimidating but the idea is very simple. It simply states that the value of your starting point is the immediate reward you expect to get from being there + the discounted value of where you land next. There is a more formal explanation and proof of the Bellman equation [here](https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/). 

### Q-Learning

Now we can learn <img src="http://latex.codecogs.com/svg.latex?Q^*"/> by using this definition and dynamic programming.

Let <img src="http://latex.codecogs.com/svg.latex?Q" /> be an action-value function which hopefully approximates <img src="http://latex.codecogs.com/svg.latex?Q^*" />. Initially it can be completely random or all zero, but it will hopefully converge to <img src="http://latex.codecogs.com/svg.latex?Q^*" />. For now, let's assume that we have a **tabular representation** of <img src="http://latex.codecogs.com/svg.latex?Q" />, this means that there is a large table with two columns. The left column contains all of the state-action action pairs, the right column has the <img src="http://latex.codecogs.com/svg.latex?Q" />-value for each state-action pair.

The **Bellman Error** is the update to our expected return when we observe the next state <img src="http://latex.codecogs.com/svg.latex?s^'" />

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\underbrace{r(s,&space;a)&plus;\gamma&space;\max&space;_{a}&space;Q\left(s^{\prime},&space;a\right)}_{\text&space;{inside&space;}&space;\mathbb{E}&space;\text&space;{&space;in&space;the&space;Bellman&space;eqn&space;}}-Q(s,&space;a)">
</p>

The left side is our target, and the right side is our current prediction. We want to minimize this error in our <img src="http://latex.codecogs.com/svg.latex?Q" />-function so that it becomes optimal, this is what the <img src="http://latex.codecogs.com/svg.latex?Q" />-learning algorithm does. At each timestep, we use the environment to sample consecutive states and actions <img src="http://latex.codecogs.com/svg.latex?(s,a,s',r)" />. Then update our <img src="http://latex.codecogs.com/svg.latex?Q" />-function to optimize it.
 
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?Q(s,a)\leftarrow&space;Q(s,a)&plus;\alpha\underbrace{\left[r(s,a)&plus;\gamma\max_{a^{\prime}}&space;Q\left(s^{\prime},a^{\prime}\right)-Q(s,a)\right]}_{\text{Bellman&space;error}}"/>
</p>

[Here](https://www.mladdict.com/q-learning-simulator) is a very good demonstration of the Q-learning algorithm in action.

### Exploration-Exploitation Tradeoff

One of the problems with <img src="http://latex.codecogs.com/svg.latex?Q" />-learning is that it only learns about states and actions it visits. If it never visits a state, it will never understand the value of going to that state. So sometimes we should pick suboptimal actions in order to visit new states.

There is a tradeoff between exploring too much or too little. This is called the **exploration-exploitation tradeoff**. If we always exploit what we think is good, we will never know what is actually good. If we always explore our environment, we will fail very early on, so we cannot get good scores nor explore advanced states.

A very simple solution is to use the **<img src="http://latex.codecogs.com/svg.latex?\mathbf{\epsilon}" />-greedy strategy**. <img src="http://latex.codecogs.com/svg.latex?\epsilon" /> is a hyperparameter that we choose, it represents how much we want to explore. With a probability of <img src="http://latex.codecogs.com/svg.latex?\epsilon" />, we choose a random action to explore the environment. With a probability of 1-<img src="http://latex.codecogs.com/svg.latex?\epsilon" />, we exploit by choosing the optimal action according to <img src="http://latex.codecogs.com/svg.latex?Q" />

### Q-Function Approximation using Neural Networks

Let's fix our previous assumption about a tabular representation of <img src="http://latex.codecogs.com/svg.latex?Q" />. There is nothing inherently wrong with a tabular representation of <img src="http://latex.codecogs.com/svg.latex?Q" />, but it is only feasible for problems with very small state space. Breakout has a very large state space.

Atari has a 210 x 160 screen with 128 possible colors for each pixel. If we stack the last 4 frames as input to our state (to satisfy Markov assumption), then the state space has a size of <img src="http://latex.codecogs.com/svg.latex?128^{4 \cdot 210 \cdot 160}" />. Even if we use ram inputs, the state space still has a size of <img src="http://latex.codecogs.com/svg.latex?256^{128}" />. It is simply impossible to maintain a table this large.

This is where neural networks and deep learning come in. Neural networks are a universal approximator that can extract higher level features (e.g. the network can understand that colors dowi not matter, a nearly completed game state should have a lower <img src="http://latex.codecogs.com/svg.latex?Q" />-value than the initial state, etc). Instead of updating the <img src="http://latex.codecogs.com/svg.latex?Q" />-value of a single state one at a time, we update our parameters for our <img src="http://latex.codecogs.com/svg.latex?Q" />-network so that the <img src="http://latex.codecogs.com/svg.latex?Q" />-network will give a better prediction for all states on average. It doesn't have to memorize the <img src="http://latex.codecogs.com/svg.latex?Q" />-value for each possible state-action pair.

A common architecture is for the neural network to take an entire state as its input, the output layer contains the <img src="http://latex.codecogs.com/svg.latex?Q" />-value for each action at that state. 

<p align="center">
<img src="https://deeplizard.com/images/Deep%20Q-Network.png", height="300"/>
</p>

Just like before, we update our network using our experience <img src="http://latex.codecogs.com/svg.latex?(s,a,s',r)" /> interacting with the environment. We want our network's prediction <img src="http://latex.codecogs.com/svg.latex?Q(s,a)" /> to move towards the target <img src="http://latex.codecogs.com/svg.latex?t" />.

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\begin{array}{c}{t=r(s, a)+\gamma \underset{a'}{\text{max}}Q\left(s^{\prime},a^{\prime}\right)}\\ {\theta \leftarrow\theta-\alpha\frac{\partial}{\partial\theta}\mathcal{L}(Q(s,a),t)}\end{array}"/>
</p>

## Results

<p align="center">
<img src="https://i.imgur.com/wFZmjtj.gif" />
</p>
This is the best game played by my agent, it scored 376 points. 

<p align="center">
<img src="https://i.imgur.com/GqtJ0qp.png" width="400" /> <img src="https://i.imgur.com/E5p0UBN.png" width="400"/>
</p>

These are the scores achieved by the agent after training for 190 hours. The blue lines are the score for each episode, the orange lines are the 100 episode rolling average. Both graphs depict the same data, but the second graph is zoomed in so it is easier to see the improvement. It peaked at 18000 episodes, then the performance starts to fluctuate randomly. 

### Model Architecture and Hyperparameters

My model has 7 hidden layers, with 512, 256, 128, 64, 32, 16, and 8 hidden units for each respective layer, ReLU activation is used after each layer. There are a total of 241,308 weights in this network. 

<img src="http://latex.codecogs.com/svg.latex?\epsilon" /> starts at 1, then reduced linearly to 0.1 over 1,000,000 timesteps.
<img src="http://latex.codecogs.com/svg.latex?\gamma" /> is set to 0.99, batch size set to 32, and learning rate set to 0.00001.

## Optimizations

Even though my agent did not perform very well, here are some of the optimizations I made to improve the performance of my agent.

### Huber Loss

Usually, Mean Square Error or Cross Entropy loss is used to measure the error of a prediction. In reinforcement learning, a different loss function called  **Huber loss** is commonly used instead.

Huber loss (called smooth l1 loss in PyTorch) is a mixture of Mean Absolute Error (L1 loss) and Mean Square Error (L2 loss). The benefit of Huber loss is that it provides the benefits of MSE (gradient increases with error) when the error is small but it is less sensitive to outliers when the error is high. The Huber loss is a piecewise function composed of rescaled MSE and translated MAE.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?z_{i}=\left\{\begin{array}{ll}{0.5\left(x_{i}-y_{i}\right)^{2},}&{\text{&space;if&space;}\left|x_{i}-y_{i}\right|<1}\\&space;{\left|x_{i}-y_{i}\right|-0.5,}&{\text{&space;otherwise}}\end{array}\right."/>
</p>

<p align="center">
<img src="https://i.imgur.com/wBAEdJn.png" height="250"/>
</p>

Here is the comparison between MSE (blue) and Huber (red). MSE's gradient increases quadratically, Huber's gradient increases with error when the error is small but remains constant once the error exceeds 1.

### Experience Replay

Experience replay is a biologically inspired mechanism proposed by Lin<sup><a name="secondb">[\[2\]](#seconda)</a></sup>.  In real life, humans and animals do not just learn from their experience once and then discard that experience. They can remember and reuse experiences from the past multiple times.

The algorithm for Experience Replay is very simple. Instead of learning directly from the interaction with the environment, we store our transition <img src="http://latex.codecogs.com/svg.latex?(s,a,s',r)" /> in a large table, then sample transitions uniformly from the table when learning. Learning usually occurs after an interaction.

Most deep learning algorithms assume that data samples are independent, but reinforcement learning typically encounters sequences of highly correlated state, sampling our experience uniformly from the table can break the correlation.

### Prioritized Experience Replay

Can we do better than uniform sampling? Schaul et al suggest that an agent can learn more effectively from some transitions than from others<sup> <a name="thirdb"> [\[3\]](#thirda)</a> </sup>. Some transitions are redundant but some can provide very useful information. Some transitions may not be immediately useful to the agent but might become so when the agent competence increases.

The magnitude of the Bellman error is used to prioritize the transitions. A transition with higher Bellman error is more "surprising", so we should learn from those transitions first. This prioritization can lead to a loss of diversity, which is alleviated with stochastic prioritization. It also introduces bias but is corrected with importance sampling. In this project, I used someone else's implementation for prioritized experience replay <sup> <a name="fourthb"> [\[4\]](#fourtha)</a> </sup>.

### Fixed Target Networks

Let's examine the updates for our <img src="http://latex.codecogs.com/svg.latex?Q" />-network more closely

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\begin{array}{c}{t=r(s, a)+\gamma \underset{a'}{\text{max}}Q\left(s^{\prime},a^{\prime}\right)}\\ {\theta \leftarrow\theta-\alpha\frac{\partial}{\partial\theta}\mathcal{L}(Q(s,a),t)}\end{array}"/>
</p>

Remember, we are updating our <img src="http://latex.codecogs.com/svg.latex?Q" />-network so <img src="http://latex.codecogs.com/svg.latex?Q(s,a)" /> moves towards the target. However, the <img src="http://latex.codecogs.com/svg.latex?Q" />-network is used twice, to calculate the <img src="http://latex.codecogs.com/svg.latex?Q" />-value and the target. So the <img src="http://latex.codecogs.com/svg.latex?Q" />-value is moving closer to the target, but the target is also moving away from the <img src="http://latex.codecogs.com/svg.latex?Q" />-value. This causes the network to be unstable during training.

This can be solved by using a fixed target network. The target network is a copy of the <img src="http://latex.codecogs.com/svg.latex?Q" />-network used solely to calculate the target. When the networks are initialized, the weights of the <img src="http://latex.codecogs.com/svg.latex?Q" />-network is coped to the target network. During training, we only update the <img src="http://latex.codecogs.com/svg.latex?Q" />-network, the target network stays constant but is periodically (every 10000 interactions for this project) updated by copying the weights from the <img src="http://latex.codecogs.com/svg.latex?Q" />-network to the target network.

### Double DQN

Let's take a closer look at the way <img src="http://latex.codecogs.com/svg.latex?Q" />-targets are calculated.

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?t=r(s,a)&plus;\gamma\max_{a'}Q\left(s',a'\right)"/>
</p>

In this equation, we are assuming that the best action for the next state is the action with the highest <img src="http://latex.codecogs.com/svg.latex?Q" />-value, but how are we sure this is true? This would be true if we already have the optimal <img src="http://latex.codecogs.com/svg.latex?Q" />-function, but we do not have the optimal <img src="http://latex.codecogs.com/svg.latex?Q" />-function yet, we are trying to learn it.
   
The accuracy of the <img src="http://latex.codecogs.com/svg.latex?Q" />-values depends on what states/actions we have explored. When we visit a state for the first time, we do not have enough information about the best action to take. If non-optimal actions are regularly given a higher Q value than the optimal best action, learning will be complicated.

Hasselt et al proposed double DQNs to decouple action selection from target evaluation<sup> <a name="fifthb"> [\[5\]](#fiftha)</a> </sup>. We use the <img src="http://latex.codecogs.com/svg.latex?Q" />-network to select the best action to take for the next state, then use the target network to evaluate the <img src="http://latex.codecogs.com/svg.latex?Q" />-value of taking that action. Doing this can help reduce overestimation of <img src="http://latex.codecogs.com/svg.latex?Q" />-values.
<p align="center">
<img src="https://cdn-media-1.freecodecamp.org/images/1*g5l4q162gDRZAAsFWtX7Nw.png" width="400" />
</p>

## References
<a name="firsta">[1](#firstb)</a>: https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii 

<a name="seconda">[2](#secondb)</a>: http://www.incompleteideas.net/lin-92.pdf

<a name="thirda">[3](#thirdb)</a>: https://arxiv.org/abs/1511.05952

<a name="fourtha">[4](#fourthb)</a>: https://github.com/rlcode/per/

<a name="fiftha">[5](#fifthb)</a>: https://arxiv.org/abs/1509.06461
