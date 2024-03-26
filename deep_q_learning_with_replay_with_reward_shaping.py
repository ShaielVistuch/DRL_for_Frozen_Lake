import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ReplayMemory import ReplayMemory
class deep_q_learning_with_replay_with_reward_shaping():
    def __init__(self, env, siz, batch_size=16):
        # Exploration \ Exploitation parameters
        self.epsilon = 1.0  # Exploration parameter
        self.max_epsilon = 1.0  # Max for exploration
        self.min_epsilon = 0.01  # Min for exploration
        self.decay_rate = 0.0001  # Exponential decay factor
        self.Transition =namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        # Define the Q-network
        class QNetwork(nn.Module):
            def __init__(self, n_observations, n_actions):
                super(QNetwork, self).__init__()
                self.layer1 = nn.Linear(n_observations, 16)
                #self.layer1.weight.data.fill_(0)
                self.layer2 = nn.Linear(16, 16)
                #self.layer2.weight.data.fill_(0)
                self.layer3 = nn.Linear(16, 4)
                #self.layer3.weight.data.fill_(0)
                self.layer4 = nn.Linear(4, n_actions)
                #self.layer4.weight.data.fill_(0)

            # Returns tensor
            def forward(self, x):
                x = F.relu(self.layer1(x))
                x = F.relu(self.layer2(x))
                x = F.relu(self.layer3(x))
                return self.layer4(x)

        # Initialize the environment
        self.size = siz
        self.env = env
        self.state_size = self.size * self.size
        self.action_size = (self.env.get_possible_actions()).size

        # Initialize the Q-network, loss function, and optimizer
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        # Training parameters
        self.num_episodes = 30000
        self.gamma = 0.8
        self.epsilon = 1
        #self.doneCount = 0

        # Reward list (for the Learning Curve plot)
        self.rewards_list = []

        # Create a replay memory buffer
        self.BATCH_SIZE = batch_size
        self.memory = ReplayMemory(30000)


    def train(self, num_episodes):
        for episode in range(num_episodes):

            #print("Episode number: ", episode)

            # Getting the state -> remember that the state is an integer
            stateIndex = self.env.reset()
            new_state_arr = np.zeros(self.state_size)
            new_state_arr[stateIndex] = 1
            # Turing the state from an int to a pytorch tensor
            state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)

            done = False
            max_step_per_training_ep = 100
            total_reward_for_ep = 0
            step_number = 0
            while (not done) and (step_number < max_step_per_training_ep):
            #while not done:
                step_number += 1

                random_num = random.uniform(0, 1)
                # epsilon greedy policy
                if random_num > self.epsilon:
                    with torch.no_grad():
                        action = self.q_network(state).argmax()
                        action = torch.tensor([[action]])
                else:
                    # works print(torch.tensor([[env.get_random_action().value]], device=device, dtype=torch.long))
                    action = torch.tensor([[self.env.get_random_action().value]], dtype=torch.long)


                next_state, reward, done = self.env.stepWithRewardShaping(action.item())

                new_state_arr = np.zeros(self.state_size)
                new_state_arr[next_state] = 1
                next_state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32)

                self.memory.push(state, action, next_state, reward)

                if len(self.memory) >= 4*self.BATCH_SIZE:


                    self.epsilon = (self.max_epsilon - self.min_epsilon) * np.exp(
                        -self.decay_rate * episode) + self.min_epsilon

                    Transition, transitions = self.memory.sample(self.BATCH_SIZE)
                    # Transpose the batch
                    batch = self.Transition(*zip(*transitions))

                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                            batch.next_state)), dtype=torch.bool)# device=device,
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                       if s is not None])
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)

                    state_action_values = self.q_network(state_batch).gather(1, action_batch)
                    next_state_values = torch.zeros(self.BATCH_SIZE)#, device=device
                    with torch.no_grad():
                        next_state_values[non_final_mask] = self.q_network(non_final_next_states).max(1).values

                    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

                    # Calculating loss
                    loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                    i = 0
                    while i < (2*self.BATCH_SIZE ):
                        self.memory.remove()
                        i = i +1



                state = next_state
                total_reward_for_ep += reward.item()

            if episode % 1000 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward_for_ep}")

            self.rewards_list.append(total_reward_for_ep)

    def plot_learning_curve(self):
        # Plotting the learning curve
        x = [x for x in range(len(self.rewards_list))]
        plt.figure()
        plt.plot(x, self.rewards_list, '-b', label='Reward')
        plt.title("Learning Curve\nDEEP Q-ALGO\nwith replay buffer\nwith reward shaping")
        plt.xlabel("Episode #")
        plt.ylabel("Reward per episode")

        # Take 'window' episode averages and plot them too
        data = np.array(self.rewards_list)
        window = 10
        average_y = []
        for ind in range(len(data) - window + 1):
            average_y.append(np.mean(data[ind:ind + window]))
        for ind in range(window - 1):
            average_y.insert(0, np.nan)
        plt.plot(x, average_y, 'r.-', label='10-episode average')
        plt.legend()
        plt.show(block=True)
