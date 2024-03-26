from action import Action
import numpy as np
from copy import deepcopy
from numpy.random import seed
import random


def addRewardShape(s, current_state_index, size):
    r_s, c_s = s //size, s%size
    r_ns,c_ns =current_state_index // size, current_state_index % size
    change = (r_ns -r_s) + (c_ns -c_s)
    return change


class Frozen_Lake_Environment:

    def __init__(self, size):
        self.current_state_index = 0
        # Initializing last state to be the Goal ('G')
        self.board = ['S']  # The board is used to store the states' type
        self.SIZE = size  # Length\width of the square-shaped board
        self.hole_index_list = []
        self.state_space = []
        self.possible_actions = np.array([Action.Left, Action.Right, Action.Up, Action.Down])

        # Initializing all states except the last one to be Frozen ('F')
        for i in range(1, self.SIZE * self.SIZE - 1):
            self.board.append('F')

        # Initializing last state to be the Goal ('G')
        self.board.append('G')

        for i in range(np.array(self.board).size):
            self.state_space.append(i)

    def print_environment_parameters(self):
        print("-----------self.SIZE:-------------")
        print(self.SIZE)
        print("-------------self.board:------------")
        print(self.board)
        print("---------self.hole_index_list:---------")
        print(self.hole_index_list)
        print("---------self.state_space:--------")
        print(self.state_space)
        print("--------self.possible_actions:--------")
        print(self.possible_actions)

    def add_hard_restrictions_to_env(self):
        for i in range(self.SIZE * self.SIZE - 1):
            self.board[i] = 'H'
            self.hole_index_list.append(i)
        for i in range(self.SIZE * self.SIZE - 1):
            if ((i + 1) % self.SIZE) == 0:
                self.board[i] = 'F'
            if i < (self.SIZE - 1):  # First row doesn't have holes
                self.board[i] = 'F'

        self.board[0] = 'S'  # Upper-Left corner is reserved for Start
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def add_easy_restrictions_to_env(self):
        for i in range(self.SIZE - 1):
            rand = random.choice(self.state_space)
            self.board[rand] = 'H'
            self.hole_index_list.append(rand)

        self.board[0] = 'S'  # First corner is reserved for Start
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def add_hole_layout1(self):
        self.hole_index_list.append(1)
        self.board[1] = 'H'
        self.hole_index_list.append(21)
        self.board[21] = 'H'
        self.hole_index_list.append(22)
        self.board[22] = 'H'

        self.board[0] = 'S'  # First corner is reserved for Start
        self.board[self.SIZE * self.SIZE - 1] = 'G'  # Last corner is reserved for Goal

    def step(self, action_index):
        action = Action(action_index)

        # Checking if the action chosen is within the defined board
        if self.invalid_action(action):
            return self.current_state_index, -2 * self.SIZE / 5, False

        # Updating current state
        if action == Action.Left:
            self.current_state_index -= 1
        elif action == Action.Right:
            self.current_state_index += 1
        elif action == Action.Up:
            self.current_state_index -= self.SIZE
        else:
            self.current_state_index += self.SIZE

        state_type = self.board[self.current_state_index]

        # We reached the starting point. We need to continue -> returning False
        # if state_type == 'S':
        #     return self.current_state_index, -10 * self.SIZE * self.SIZE, False

        # We reached a frozen place. We need to continue -> returning False
        if state_type == 'F' or state_type == 'S':
            return self.current_state_index, -self.SIZE / 50, False

        # We reached the GOAL! -> Returning True
        elif state_type == 'G':
            return self.current_state_index, self.SIZE, True

        # We reached a HOLE! -> Returning True
        else:
            return self.current_state_index, -self.SIZE / 5, True

    def stepWithRewardShaping(self, action_index):
        action = Action(action_index)
        s = self.current_state_index
        # Checking if the action chosen is within the defined board
        if self.invalid_action(action):
            return self.current_state_index, -2 * self.SIZE / 5, False


        # Updating current state
        if action == Action.Left:
            self.current_state_index -= 1
        elif action == Action.Right:
            self.current_state_index += 1
        elif action == Action.Up:
            self.current_state_index -= self.SIZE
        else:
            self.current_state_index += self.SIZE

        rewardShape = addRewardShape(s, self.current_state_index,self.SIZE)

        #print(action, s, self.current_state_index)
        state_type = self.board[self.current_state_index]

        # We reached the starting point. We need to continue -> returning False
        # if state_type == 'S':
        #     return self.current_state_index, -10 * self.SIZE * self.SIZE, False

        # We reached a frozen place. We need to continue -> returning False
        if state_type == 'F' or state_type == 'S':
            rewardShape = rewardShape -self.SIZE / 50
            #print('F ', rewardShape)
            return self.current_state_index, rewardShape, False

        # We reached the GOAL! -> Returning True
        elif state_type == 'G':
            rewardShape = rewardShape + self.SIZE
            #print('G ', rewardShape)
            return self.current_state_index,rewardShape , True

        # We reached a HOLE! -> Returning True
        else:
            rewardShape = rewardShape -self.SIZE / 5
            #print('H ', rewardShape)
            return self.current_state_index, rewardShape, True


    def invalid_action(self, action):
        if action == Action.Left and self.current_state_index % self.SIZE == 0:
            return True
        elif action == Action.Right and self.current_state_index % self.SIZE == (self.SIZE - 1):
            return True
        elif action == Action.Up and self.current_state_index < self.SIZE:
            return True
        elif (action == Action.Down and (
                self.SIZE * self.SIZE - self.SIZE) <= self.current_state_index <= self.SIZE * self.SIZE - 1):
            return True
        else:
            return False

    def print_on_board_current_state(self):
        temp_board = deepcopy(self.board)
        for i in range(0, self.SIZE * self.SIZE):
            if self.current_state_index != i:
                if temp_board[i] == 'F':
                    print('+', end=" ")
                if temp_board[i] == 'H':
                    print('h', end=" ")
                if temp_board[i] == 'G':
                    print('G', end=" ")
                if temp_board[i] == 'S':
                    print('S', end=" ")
            else:
                print('A', end=" ")
            if i % self.SIZE == (self.SIZE - 1):
                print()

    def reset(self):
        self.current_state_index = 0
        return self.current_state_index

    def get_holes(self):
        return self.hole_index_list

    def get_possible_actions(self):
        return self.possible_actions

    def get_state_space(self):
        return self.state_space

    def get_random_action(self):
        return np.random.choice(self.possible_actions)
