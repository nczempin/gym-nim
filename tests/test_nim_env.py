import pytest
import numpy as np
import gymnasium as gym
import gym_nim


class TestNimEnv:
    """Test suite for the Nim environment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env = gym.make('nim-v0')
        # Access unwrapped environment for custom methods
        self.unwrapped = self.env.unwrapped
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_environment_creation(self):
        """Test that the environment can be created."""
        assert self.env is not None
        assert hasattr(self.env, 'action_space')
        assert hasattr(self.env, 'observation_space')
    
    def test_reset(self):
        """Test environment reset."""
        state, info = self.env.reset()
        assert state is not None
        assert 'board' in state
        assert 'on_move' in state
        np.testing.assert_array_equal(state['board'], [7, 5, 3])
        assert state['on_move'] == 1
        assert isinstance(info, dict)
    
    def test_valid_move(self):
        """Test making a valid move."""
        self.env.reset()
        # Take 2 pieces from pile 0
        state, reward, terminated, truncated, info = self.env.step([0, 2])
        
        assert state['board'][0] == 5  # 7 - 2 = 5
        assert state['on_move'] == 2  # Player switched
        assert reward == 0  # No reward for normal move
        assert not terminated  # Game continues
        assert not truncated  # Not truncated
    
    def test_invalid_pile_index(self):
        """Test move with invalid pile index."""
        self.env.reset()
        # Try to access non-existent pile
        state, reward, terminated, truncated, info = self.env.step([3, 1])
        
        assert reward == -2  # Penalty for illegal move
        assert terminated  # Game ends on illegal move
    
    def test_take_too_many_pieces(self):
        """Test trying to take more pieces than available."""
        self.env.reset()
        # Try to take 4 pieces from pile 2 (which has only 3)
        state, reward, terminated, truncated, info = self.env.step([2, 4])
        
        assert reward == -2  # Penalty for illegal move
        assert terminated  # Game ends on illegal move
    
    def test_game_end_condition(self):
        """Test that game ends when all piles are empty."""
        self.env.reset()
        # Empty all piles
        moves = [
            [0, 3], [1, 3], [2, 2],  # Player 1 moves
            [0, 3], [1, 2], [2, 1],  # Player 2 moves
            [0, 1]  # Player 1 takes last piece
        ]
        
        for i, move in enumerate(moves[:-1]):
            state, reward, terminated, truncated, info = self.env.step(move)
            assert not terminated, f"Game ended early at move {i}"
        
        # Last move should end the game
        state, reward, terminated, truncated, info = self.env.step(moves[-1])
        assert terminated
        assert reward == -1  # Player who takes last piece loses
        assert all(count == 0 for count in state['board'])
    
    def test_move_generator(self):
        """Test legal move generation."""
        self.env.reset()
        moves = self.unwrapped.move_generator()
        
        # Should have 9 moves initially: 3+3+3
        assert len(moves) == 9
        
        # Check specific moves are included
        assert [0, 1] in moves  # Take 1 from pile 0
        assert [0, 2] in moves  # Take 2 from pile 0
        assert [0, 3] in moves  # Take 3 from pile 0
        assert [1, 1] in moves  # Take 1 from pile 1
        assert [2, 3] in moves  # Take 3 from pile 2
    
    def test_move_generator_partial_board(self):
        """Test move generation with partially empty board."""
        self.env.reset()
        # Empty first pile
        self.env.step([0, 3])
        self.env.step([0, 3])
        self.env.step([0, 1])
        
        moves = self.unwrapped.move_generator()
        # Should not include moves from empty pile 0
        assert not any(move[0] == 0 for move in moves)
    
    def test_set_board(self):
        """Test setting custom board configuration."""
        self.env.reset()
        self.unwrapped.set_board([1, 2, 3])
        
        state = self.unwrapped.state
        np.testing.assert_array_equal(state['board'], [1, 2, 3])
    
    def test_render(self):
        """Test that render doesn't crash."""
        self.env.reset()
        # Should not raise an exception
        self.env.render()
    
    def test_step_before_reset(self):
        """Test that stepping before reset raises an error."""
        new_env = gym.make('nim-v0')
        # Gymnasium raises ResetNeeded for stepping before reset
        with pytest.raises(Exception):  # Could be ResetNeeded or ValueError
            new_env.step([0, 1])
        new_env.close()
    
    def test_invalid_action_format(self):
        """Test various invalid action formats."""
        self.env.reset()
        
        # Test None action
        with pytest.raises(ValueError, match="Invalid action format"):
            self.env.step(None)
        
        # Test wrong length action
        with pytest.raises(ValueError, match="Invalid action format"):
            self.env.step([0])
        
        # Test action with too many elements
        with pytest.raises(ValueError, match="Invalid action format"):
            self.env.step([0, 1, 2])
    
    def test_player_alternation(self):
        """Test that players alternate correctly."""
        self.env.reset()
        assert self.unwrapped.state['on_move'] == 1
        
        self.env.step([0, 1])
        assert self.unwrapped.state['on_move'] == 2
        
        self.env.step([1, 1])
        assert self.unwrapped.state['on_move'] == 1

    def test_zero_count_action(self):
        """Test that taking 0 pieces is invalid."""
        self.env.reset()
        state, reward, terminated, truncated, info = self.env.step([0, 0])
        
        assert reward == -2  # Penalty for illegal move
        assert terminated  # Game ends on illegal move

    def test_negative_count_action(self):
        """Test that taking negative pieces is invalid."""
        self.env.reset()
        state, reward, terminated, truncated, info = self.env.step([0, -1])
        
        assert reward == -2  # Penalty for illegal move
        assert terminated  # Game ends on illegal move

    def test_negative_pile_index(self):
        """Test that negative pile index is invalid."""
        self.env.reset()
        state, reward, terminated, truncated, info = self.env.step([-1, 1])
        
        assert reward == -2  # Penalty for illegal move
        assert terminated  # Game ends on illegal move

    def test_state_immutability_after_illegal_move(self):
        """Test that state doesn't change after illegal move."""
        self.env.reset()
        original_board = self.unwrapped.state['board'].copy()
        original_player = self.unwrapped.state['on_move']
        
        # Make illegal move
        self.env.step([0, 10])
        
        # State should be unchanged except for game end
        np.testing.assert_array_equal(self.unwrapped.state['board'], original_board)
        assert self.unwrapped.state['on_move'] == original_player

    def test_render_without_reset(self):
        """Test rendering when game not started."""
        new_env = gym.make('nim-v0')
        
        # Gymnasium enforces reset before render, so this should raise an exception
        with pytest.raises(Exception):  # ResetNeeded or similar
            new_env.render()
        
        new_env.close()

    def test_move_generator_empty_game(self):
        """Test move generation when all piles are empty."""
        self.env.reset()
        self.unwrapped.set_board([0, 0, 0])
        moves = self.unwrapped.move_generator()
        assert len(moves) == 0

    def test_move_generator_single_piece(self):
        """Test move generation with single pieces."""
        self.env.reset()
        self.unwrapped.set_board([1, 0, 1])
        moves = self.unwrapped.move_generator()
        
        expected_moves = [[0, 1], [2, 1]]
        assert len(moves) == 2
        assert [0, 1] in moves
        assert [2, 1] in moves

    def test_set_board_before_reset(self):
        """Test setting board before reset."""
        new_env = gym.make('nim-v0')
        new_env.unwrapped.set_board([1, 2, 3])
        
        # Should initialize state if not present
        assert new_env.unwrapped.state is not None
        np.testing.assert_array_equal(new_env.unwrapped.state['board'], [1, 2, 3])
        assert new_env.unwrapped.state['on_move'] == 1
        new_env.close()

    def test_action_space_properties(self):
        """Test action and observation space properties."""
        assert hasattr(self.env.action_space, 'n')
        assert hasattr(self.env.observation_space, 'n')
        assert self.env.action_space.n == 9  # As defined in the environment
        assert self.env.observation_space.n == 8*8*8*2  # As defined in the environment
    
    def test_observation_space_dict_structure(self):
        """Test that observation space is properly structured as Dict."""
        from gymnasium import spaces
        
        # Check that observation_space is a Dict
        assert isinstance(self.env.observation_space, spaces.Dict)
        
        # Check the structure contains the expected keys
        assert 'board' in self.env.observation_space.spaces
        assert 'on_move' in self.env.observation_space.spaces
        
        # Check board space properties
        board_space = self.env.observation_space.spaces['board']
        assert isinstance(board_space, spaces.Box)
        assert board_space.shape == (3,)
        assert board_space.dtype == np.int32
        assert np.all(board_space.low == 0)
        assert np.all(board_space.high == 7)
        
        # Check on_move space properties
        on_move_space = self.env.observation_space.spaces['on_move']
        assert isinstance(on_move_space, spaces.Discrete)
        assert on_move_space.n == 3  # Discrete(3) means values 0, 1, 2
        assert on_move_space.start == 1  # But starts at 1, so valid values are 1, 2
    
    def test_observation_space_backward_compatibility(self):
        """Test backward compatibility with .n attribute."""
        # The observation space should have .n for compatibility with qtable.py
        assert hasattr(self.env.observation_space, 'n')
        assert self.env.observation_space.n == 8 * 8 * 8 * 2
        
        # Ensure the .n attribute persists after operations
        obs_space = self.env.observation_space
        assert obs_space.n == 8 * 8 * 8 * 2
    
    def test_observation_within_space(self):
        """Test that observations are within the declared observation space."""
        # Reset and check initial observation
        state, info = self.env.reset()
        
        # Manually check that state matches observation space
        # (gymnasium's passive checker might not understand our custom Dict+.n hybrid)
        assert isinstance(state, dict)
        assert 'board' in state
        assert 'on_move' in state
        
        # Check board values are within bounds
        board = state['board']
        assert isinstance(board, np.ndarray)
        assert board.shape == (3,)
        assert np.all(board >= 0)
        assert np.all(board <= 7)
        
        # Check on_move is valid
        assert state['on_move'] in [1, 2]
        
        # Make a move and check again
        state, reward, terminated, truncated, info = self.env.step([0, 2])
        assert isinstance(state, dict)
        assert np.all(state['board'] >= 0)
        assert np.all(state['board'] <= 7)
        assert state['on_move'] in [1, 2]

    def test_complete_game_scenario(self):
        """Test a complete game from start to finish."""
        self.env.reset()
        moves_made = 0
        max_moves = 50  # Safety limit
        
        while moves_made < max_moves:
            moves = self.unwrapped.move_generator()
            if not moves:
                break
                
            # Make a random valid move
            action = moves[0]  # Take first available move
            state, reward, terminated, truncated, info = self.env.step(action)
            moves_made += 1
            
            if terminated:
                # Game should end with someone losing
                assert reward == -1, f"Expected losing reward, got {reward}"
                break
                
        assert moves_made < max_moves, "Game should end within reasonable moves"

    def test_reward_structure(self):
        """Test all possible reward values."""
        self.env.reset()
        
        # Normal move should give 0 reward
        state, reward, terminated, truncated, info = self.env.step([0, 1])
        assert reward == 0
        
        # Reset for illegal move test
        self.env.reset()
        state, reward, terminated, truncated, info = self.env.step([0, 10])
        assert reward == -2  # Illegal move penalty
        
        # Test losing (this requires specific setup)
        self.env.reset()
        self.unwrapped.set_board([1, 0, 0])  # Only one piece left
        state, reward, terminated, truncated, info = self.env.step([0, 1])
        assert reward == -1  # Losing player