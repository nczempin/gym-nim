# gym-nim

An OpenAI Gym environment for the classic game of Nim, designed for multi-agent reinforcement learning experiments.

## What is Nim?

Nim is one of the oldest mathematical strategy games, dating back centuries. It's a perfect game for learning reinforcement learning because:

1. **Simple Rules**: Easy to understand but strategically deep
2. **Perfect Information**: Both players see the complete game state
3. **Mathematical Beauty**: Has an elegant solution using binary XOR
4. **Quick Games**: Episodes typically last 10-20 moves
5. **Multi-Agent Learning**: Agents can learn by playing against themselves

### Game Rules

Two players take turns removing pieces from piles:

```
Pile 0: ●●●●●●● (7 pieces)
Pile 1: ●●●●●   (5 pieces)  
Pile 2: ●●●     (3 pieces)

Player 1's turn → Remove 1-3 pieces from any ONE pile
```

The player who takes the **last piece loses** (normal play convention).

## Environment Details

- **State Space**: Dictionary with board configuration and current player
  ```python
  {'board': array([7, 5, 3]), 'on_move': 1}
  ```
- **Action Space**: `[pile_index, pieces_to_take]` where:
  - `pile_index`: 0, 1, or 2 (which pile to take from)
  - `pieces_to_take`: 1, 2, or 3 (how many to remove)
- **Rewards**:
  - `-1`: Losing the game (taking the last piece)
  - `-2`: Making an illegal move
  - `0`: All other moves
- **Players**: Alternates between Player 1 and Player 2

## Requirements

- **Python 3.9-3.12** (tested on all versions)
- **OpenAI Gym 0.9.1** (legacy version for compatibility)
- **NumPy** (for array operations)
- **pytest** (for running tests)

## Installation

```bash
# Clone the repository
git clone https://github.com/nczempin/gym-nim.git
cd gym-nim

# Install in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov
```

## Quick Start

```python
import gym
import gym_nim

# Create and reset the environment
env = gym.make('nim-v0')
state = env.reset()
print(state)  # {'board': array([7, 5, 3]), 'on_move': 1}

# Player 1 takes 2 pieces from pile 0
state, reward, done, info = env.step([0, 2])
env.render()
# Player 2's turn
# Piles: [0]:5 [1]:5 [2]:3

# Get all legal moves for current player
moves = env.move_generator()
print(f"Legal moves: {moves}")

# Play until game ends
while not done:
    # Simple strategy: take first legal move
    if moves:
        action = moves[0]
        state, reward, done, info = env.step(action)
        moves = env.move_generator()
        
print(f"Game over! Player {state['on_move']} loses with reward {reward}")
```

## Examples

The `examples/` directory contains two demonstration agents:

### 1. Random Agent (`random_nim.py`)
Shows two random agents playing against each other:
```bash
python examples/random_nim.py
```

### 2. Q-Learning Agent (`qtable.py`)
Trains an agent using tabular Q-learning through self-play:
```bash
python examples/qtable.py
```

## Development

### Running Tests

```bash
# Run all tests with coverage
python tests.py

# Or use pytest directly
pytest tests/ -v --cov=gym_nim
```

### Docker Testing

Test in an isolated environment:
```bash
docker build -t gym-nim .
docker run --rm gym-nim
```

## API Documentation

The main class `NimEnv` provides these methods:

- **`reset()`**: Start a new game
- **`step(action)`**: Take an action and get the result
- **`render()`**: Display the current game state
- **`move_generator()`**: Get all legal moves
- **`set_board(board)`**: Set a custom board position

See the [API documentation](gym_nim/envs/nim_env.py) for detailed method descriptions and examples.

## Future Enhancements

- Migration to modern Gymnasium library (planned for v0.2.0)
- Variable pile configurations
- Different game variants (misère vs normal play)
- Advanced opponent strategies for training
