import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
class vanila_deep_q_learning_linear_network():
    def __init__(self, env, siz):
        # Exploration \ Exploitation parameters
        self.epsilon = 1.0  # Exploration parameter
        self.max_epsilon = 1.0  # Max for exploration
        self.min_epsilon = 0.01  # Min for exploration
        self.decay_rate = 0.0001  # Exponential decay factor

        # Define the Q-network
        class QNetwork(nn.Module):
            def __init__(self, n_observations, n_actions):
                super(QNetwork, self).__init__()
                self.layer1 = nn.Linear(n_observations, 16)
                # self.layer1.weight.data.fill_(0)
                self.layer2 = nn.Linear(16, 16)
                # self.layer2.weight.data.fill_(0)
                self.layer3 = nn.Linear(16, 4)
                # self.layer3.weight.data.fill_(0)
                self.layer4 = nn.Linear(4, n_actions)
                # self.layer4.weight.data.fill_(0)

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
        self.q_network_init = QNetwork(self.state_size, self.action_size)
        self.q_network_init.load_state_dict(self.q_network.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        # Training parameters
        self.num_episodes = 30000
        self.gamma = 0.8
        #self.doneCount = 0

        # Reward list (for the Learning Curve plot)
        self.rewards_list = []



    def train(self, num_episodes):
        self.q_network.load_state_dict(self.q_network_init.state_dict())
        for episode in range(num_episodes):

            # Getting the state -> remember that the state is an integer
            state = self.env.reset()
            new_state_arr = np.zeros(self.state_size)
            new_state_arr[state] = 1
            # Turing the state from an int to a pytorch tensor
            state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)

            done = False
            total_reward_for_ep = 0
            step_number = 0
            max_step_per_training_ep = self.size*self.size
            while (not done) and (step_number < max_step_per_training_ep):
            #while not done:
                step_number = step_number + 1

                # Epsilon-greedy exploration implemtation
                random_num = random.uniform(0, 1)
                # Exploitation: picking max Q value
                # if (random_num > self.epsilon and self.doneCount > 5):
                if random_num > self.epsilon:
                    with torch.no_grad():
                        q_values = self.q_network(state)
                        action = torch.argmax(q_values).item()
                # Exploration: picking a random action
                else:
                    action = (self.env.get_random_action().value)

                # As time goes we need less Exploration and more Exploitation
                self.epsilon = (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode) + self.min_epsilon

                # Take the chosen action
                next_state, reward, done = self.env.step(action)

                new_state_arr = np.zeros(self.state_size)
                new_state_arr[next_state] = 1

                # Turing the state from an int to a pytorch tensor
                next_state = torch.tensor(new_state_arr, dtype=torch.float32).unsqueeze(0)

                # Turing the reward from an int to a pytorch tensor
                reward = torch.tensor(reward, dtype=torch.float32)

                # Update Q-value using the Q-learning update rule
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.q_network(next_state))

                current = (torch.max(self.q_network(state)))

                # Calculating loss
                loss = self.criterion(current, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Updating state
                state = next_state

                # Updating reward counter
                total_reward_for_ep += reward.item()

            # Print the total reward for this episode
            if episode % 1000 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward_for_ep}")

            self.rewards_list.append(total_reward_for_ep)

    def plot_learning_curve(self):
        # Plotting the learning curve
        x = [x for x in range(len(self.rewards_list))]
        plt.figure()
        plt.plot(x, self.rewards_list, '-b', label='Reward')
        plt.title("Learning Curve\nSIMPLE DEEP Q-ALGO")
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
        #plt.close()

