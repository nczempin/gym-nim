import gym
from gym import spaces
import numpy as np

class NimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(8*8*8*2) # flattened
    def _step(self, action):
        done = False
        reward = 0

        
        pile, count = action
       
        # check move legality
        board = self.state['board']
        om = self.state['on_move']
        if (pile < 0 or pile > 2): #non-existent pile TODO generalize
            print("illegal move ", action, ".")
            done = True
            reward = -2
        else:
            pile_size = board[pile]
            if (count > pile_size):
                print("illegal move  ", action, ". taking too many")
                done = True
                reward = -2
            else:
                # swap who's on move
                self.state['on_move'] = 3 - self.state['on_move'] # alternate between 1 and 2
                board[pile] -= count
                # check game over
                pieces_left = 0
                
                if (board[0] == 0 and board[1] == 0 and board[2] == 0): #TODO generalize
                    done = True
                    reward = -1

        return self.state, reward, done, {}
    def _reset(self):
        self.state = {}
        self.state['board'] = [7, 5, 3] #TODO randomize
        self.state['on_move'] = 1 #TODO randomize?
        return self.state
    
    def set_board(self, board):
        self.state['board'] = board
    def _render(self, mode='human', close=False):
        if close:
            return
        print("on move: " , self.state['on_move'])
        for i in range (3):
            print (self.state['board'][i], end=" ")
        print()
    def move_generator(self):
        moves = []
        for i in range (3):
            #TODO generalize; this is not elegant
            if (self.state['board'][i] > 0):
                m = [i, 1]
                moves.append(m)
            if (self.state['board'][i] > 1):
                m = [i, 2]
                moves.append(m)
            if (self.state['board'][i] > 2):
                m = [i, 3]
                moves.append(m)
        return moves
    
    