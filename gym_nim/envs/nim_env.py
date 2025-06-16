import gym
from gym import spaces
import numpy as np

class NimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Action space: tuple of (pile_index, pieces_to_take)
        # We'll validate actions manually since old gym doesn't support our use case well
        self.action_space = spaces.Discrete(9)  # Placeholder, we'll validate manually
        # Observation space: simplified for old gym version
        self.observation_space = spaces.Discrete(8*8*8*2)  # flattened board + player
        self.state = None
    def _step(self, action):
        if self.state is None:
            raise ValueError("Cannot step before reset()")
        
        done = False
        reward = 0
        
        # Validate action format
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError(f"Invalid action format: {action}. Expected a tuple or list of (pile, count)")
        
        pile, count = action
        # count is already 1-3 from the examples
        
        # Check move legality
        board = self.state['board']
        
        # Validate pile index
        if pile < 0 or pile >= len(board):
            print(f"Illegal move {action}: invalid pile {pile}")
            done = True
            reward = -2
        # Validate count
        elif count <= 0:
            print(f"Illegal move {action}: count must be positive")
            done = True
            reward = -2
        # Check if trying to take too many pieces
        elif count > board[pile]:
            print(f"Illegal move {action}: trying to take {count} from pile {pile} with only {board[pile]} pieces")
            done = True
            reward = -2
        else:
            # Valid move - execute it
            board[pile] -= count
            
            # Check if game is over (all piles empty)
            if all(pile_count == 0 for pile_count in board):
                done = True
                reward = -1  # Current player loses (took the last piece)
            else:
                # Game continues, switch player
                self.state['on_move'] = 3 - self.state['on_move']
        
        return self.state, reward, done, {}
    def _reset(self):
        self.state = {
            'board': np.array([7, 5, 3], dtype=np.int32),
            'on_move': 1
        }
        return self.state
    
    def set_board(self, board):
        """Set a custom board configuration."""
        if self.state is None:
            self.state = {'board': None, 'on_move': 1}
        self.state['board'] = np.array(board, dtype=np.int32)
    
    def _render(self, mode='human', close=False):
        if close:
            return
        if self.state is None:
            print("Game not started. Call reset() first.")
            return
        
        print(f"Player {self.state['on_move']}'s turn")
        print("Piles:", end=" ")
        for i, pile_count in enumerate(self.state['board']):
            print(f"[{i}]:{pile_count}", end=" ")
        print()
    def move_generator(self):
        """Generate all legal moves from current position."""
        if self.state is None:
            return []
        
        moves = []
        board = self.state['board']
        
        for pile_idx in range(len(board)):
            pile_count = board[pile_idx]
            # Can take 1 to min(3, pile_count) pieces
            for take_count in range(1, min(4, pile_count + 1)):
                moves.append([pile_idx, take_count])
        
        return moves
    
    