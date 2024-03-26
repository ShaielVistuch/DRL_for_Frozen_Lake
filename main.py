from Frozen_Lake_Environment import Frozen_Lake_Environment
from deep_q_learning_convolution_network import deep_q_learning_convolution_network
from deep_q_learning_with_replay_with_reward_shaping import deep_q_learning_with_replay_with_reward_shaping
from deep_q_learning_linear_network import deep_q_learning_linear_network

runs = ['deep_q_learning_with_replay_with_reward_shaping']

if __name__ == '__main__':
    if 'deep_q_learning_with_replay_with_reward_shaping' in runs:
        print("+------------------------------------------+")
        print("|                                          |")
        print("|       STARTING SIMPLE DEEP Q-ALGO        |")
        print("|            Replay buffer: Yes            |")
        print("|          Convolution layer: No           |")
        print("|                                          |")
        print("+------------------------------------------+")
        print()
        # Length\Width of square-shaped board
        size = 5

        env = Frozen_Lake_Environment(size)
        env.print_environment_parameters()

        env.add_hard_restrictions_to_env()
        env.print_environment_parameters()
        env.print_on_board_current_state()
        agent = deep_q_learning_with_replay_with_reward_shaping(env, size)
        agent.train(30000)
        print(agent.rewards_list)
        agent.plot_learning_curve()

    if 'deep_q_learning_linear_network' in runs:
        print("+------------------------------------------+")
        print("|                                          |")
        print("|       STARTING SIMPLE DEEP Q-ALGO        |")
        print("|            Replay buffer: No             |")
        print("|          Convolution layers: No          |")
        print("|                                          |")
        print("+------------------------------------------+")
        print()
        # Length\Width of square-shaped board
        size = 5

        env = Frozen_Lake_Environment(size)
        env.print_environment_parameters()

        env.add_easy_restrictions_to_env()
        env.print_environment_parameters()
        env.print_on_board_current_state()
        agent = deep_q_learning_linear_network(env, size)
        agent.train(100)  # This algorithm is very slow

        #print(agent.rewards_list)
        agent.plot_learning_curve()


    if 'deep_q_learning_convolution_network' in runs:
        print("+------------------------------------------+")
        print("|                                          |")
        print("|       STARTING SIMPLE DEEP Q-ALGO        |")
        print("|            Replay buffer: No             |")
        print("|          Convolution layer: Yes          |")
        print("|                                          |")
        print("+------------------------------------------+")
        print()
        # Length\Width of square-shaped board
        size = 5

        #env = Frozen_Lake_Environment_with_goal(size)
        env = Frozen_Lake_Environment(size)
        env.print_environment_parameters()

        # Converge for easy restrictions (most of the time)
        # Algorithm is slow
        env.add_easy_restrictions_to_env()
        env.print_environment_parameters()
        env.print_on_board_current_state()
        agent = deep_q_learning_convolution_network(env, size)
        agent.train(20000)  # This algorithm is very slow, and needs about 20000-30000 to converge for easy restrictions

        #print(agent.rewards_list)
        agent.plot_learning_curve()
