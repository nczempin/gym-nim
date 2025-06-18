import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NimEnv(gym.Env):
    """A Nim game environment for reinforcement learning.
    
    Nim is a mathematical strategy game where players take turns removing objects from piles.
    The player who takes the last object loses (normal play convention).
    
    This environment implements a 3-pile Nim game where players can take 1-3 pieces per turn.
    It's designed for multi-agent self-play scenarios where agents learn by playing against
    themselves or other agents.
    
    State Space
    -----------
    The state is a dictionary with two keys:
        - 'board': numpy array of integers representing pieces in each pile [pile0, pile1, pile2]
        - 'on_move': integer (1 or 2) indicating which player's turn it is
    
    Action Space
    ------------
    Actions are tuples/lists of [pile_index, pieces_to_take] where:
        - pile_index: integer in range [0, 2] selecting which pile
        - pieces_to_take: integer in range [1, 3] for how many pieces to remove
    
    The action space is nominally Discrete(9) but actions are validated as tuples.
    
    Rewards
    -------
        - -1: for losing the game (taking the last piece)
        - -2: for making an illegal move
        - 0: for all other moves
    
    Example
    -------
    >>> env = gym.make('nim-v0')
    >>> state = env.reset()
    >>> print(state)
    {'board': array([7, 5, 3]), 'on_move': 1}
    >>> # Player 1 takes 2 pieces from pile 0
    >>> state, reward, done, info = env.step([0, 2])
    >>> print(state['board'])
    [5 5 3]
    >>> print(state['on_move'])
    2
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Action space: tuple of (pile_index, pieces_to_take)
        # We'll validate actions manually since gym doesn't support tuple actions directly
        self.action_space = spaces.Discrete(9)  # Placeholder, we'll validate manually
        
        # Observation space: Dict with board state and current player
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=7, shape=(3,), dtype=np.int32),
            'on_move': spaces.Discrete(3, start=1)  # Player 1 or 2
        })
        
        # For backward compatibility with Q-learning example that expects .n attribute
        # This represents the flattened state space size
        self.observation_space.n = 8 * 8 * 8 * 2  # All possible board states * players
        
        self.state = None
    def step(self, action):
        """Execute one time step within the environment.
        
        Parameters
        ----------
        action : tuple or list
            A two-element sequence [pile_index, pieces_to_take] where:
            - pile_index: int in [0, 2] selecting which pile
            - pieces_to_take: int in [1, 3] for how many pieces to remove
        
        Returns
        -------
        state : dict
            The new state after taking the action, with keys:
            - 'board': numpy array of current pile sizes
            - 'on_move': int (1 or 2) for current player
        reward : float
            -1 if current player loses, -2 for illegal move, 0 otherwise
        done : bool
            True if the game has ended (win/loss or illegal move)
        info : dict
            Empty dictionary for compatibility
        
        Raises
        ------
        ValueError
            If called before reset() or with invalid action format
        
        Examples
        --------
        >>> env.reset()
        >>> state, reward, done, info = env.step([0, 2])  # Take 2 from pile 0
        >>> state, reward, done, info = env.step([1, 3])  # Take 3 from pile 1
        """
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
        
        # In gymnasium: (observation, reward, terminated, truncated, info)
        return self.state, reward, done, False, {}
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state.
        
        Returns
        -------
        state : dict
            Initial state with keys:
            - 'board': numpy array [7, 5, 3] representing starting piles
            - 'on_move': int value 1 (player 1 starts)
        
        Examples
        --------
        >>> state = env.reset()
        >>> print(state)
        {'board': array([7, 5, 3]), 'on_move': 1}
        """
        self.state = {
            'board': np.array([7, 5, 3], dtype=np.int32),
            'on_move': 1
        }
        # In gymnasium: reset returns (observation, info)
        return self.state, {}
    
    def set_board(self, board):
        """Set a custom board configuration.
        
        Parameters
        ----------
        board : array-like
            List or array of integers representing pieces in each pile.
            Must have exactly 3 elements for the 3 piles.
        
        Examples
        --------
        >>> env.reset()
        >>> env.set_board([1, 0, 1])  # Only 1 piece in piles 0 and 2
        >>> print(env.state['board'])
        [1 0 1]
        """
        if self.state is None:
            self.state = {'board': None, 'on_move': 1}
        self.state['board'] = np.array(board, dtype=np.int32)
    
    def render(self):
        """Render the current game state to the console.
        
        Parameters
        ----------
        mode : str, optional
            Rendering mode. Only 'human' is supported (default).
        close : bool, optional
            If True, close the rendering window (no-op for console).
        
        Examples
        --------
        >>> env.reset()
        >>> env.render()
        Player 1's turn
        Piles: [0]:7 [1]:5 [2]:3
        """
        if self.state is None:
            print("Game not started. Call reset() first.")
            return
        
        print(f"Player {self.state['on_move']}'s turn")
        print("Piles:", end=" ")
        for i, pile_count in enumerate(self.state['board']):
            print(f"[{i}]:{pile_count}", end=" ")
        print()
    def move_generator(self):
        """Generate all legal moves from current position.
        
        Returns
        -------
        moves : list of lists
            List of all legal moves, where each move is [pile_index, pieces_to_take].
            Returns empty list if no moves available or game not started.
        
        Examples
        --------
        >>> env.reset()
        >>> moves = env.move_generator()
        >>> print(len(moves))  # Should have moves for all non-empty piles
        15
        >>> env.set_board([1, 0, 2])
        >>> moves = env.move_generator()
        >>> print(moves)
        [[0, 1], [2, 1], [2, 2]]
        """
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
    
    