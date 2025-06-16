# gym-nim

An OpenAI Gym environment for the classic game of Nim, designed for multi-agent reinforcement learning experiments.

## Game Rules

Nim is a mathematical strategy game where two players take turns removing objects from distinct piles:

- **Setup**: The game starts with 3 piles containing 7, 5, and 3 pieces respectively
- **Gameplay**: Players take turns removing 1-3 pieces from a single pile
- **Winning**: The player who takes the last piece **loses** (normal play convention)
- **Strategy**: Nim has a complete mathematical solution based on the XOR of pile sizes

## Environment Details

This implementation provides a two-player Nim environment where agents can learn strategies through self-play. Key features:

- **Players**: Two players alternate turns (Player 1 and Player 2)
- **Actions**: Choose a pile (0-2) and number of pieces to remove (1-3)
- **Rewards**: 
  - `-1` for the player who loses (takes the last piece)
  - `-2` for illegal moves (invalid pile or count)
  - `0` for all other moves
- **Observation**: Current board state and which player's turn it is

## Installation

```bash
# Clone the repository
git clone https://github.com/nczempin/gym-nim.git
cd gym-nim

# Install in development mode
pip install -e .
```

## Usage

```python
import gym
import gym_nim

# Create the environment
env = gym.make('nim-v0')

# Reset to start a new game
state = env.reset()

# Take an action (pile_index, pieces_to_take - 1)
# For example, take 2 pieces from pile 0:
action = [0, 1]  # 1 in action space means take 2 pieces
state, reward, done, info = env.step(action)

# Render the current game state
env.render()

# Get all legal moves
legal_moves = env.move_generator()
```

## Docker-based Testing

A `Dockerfile` is provided to test the project with the legacy `gym==0.9.1` dependency:

```bash
docker build -t gym-nim .
docker run --rm gym-nim
```

## Examples

See the `examples/` directory for sample agents:
- `random_nim.py`: Demonstrates random play between two agents
- `qtable.py`: Q-learning implementation for training an agent
