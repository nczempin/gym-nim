# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a custom OpenAI Gym environment implementing the game of Nim, designed for reinforcement learning experiments with self-play. The project uses the legacy `gym==0.9.1` API and provides a simple multi-agent environment where agents can learn by playing against themselves.

## Development Commands

### Setup and Installation
```bash
# Install in development mode
pip install -e .

# Install with test dependencies
pip install pytest pytest-cov
pip install -e .
```

### Running Tests
```bash
# Run tests directly
python tests.py

# Run tests with pytest and coverage
pytest tests/ -v --cov=gym_nim --cov-report=term-missing --cov-report=xml
```

### Docker-based Testing
```bash
# Build and run tests in Docker (tests with Python 3.9 and pytest)
docker build -t gym-nim .
docker run --rm gym-nim
```

### Running Examples
```bash
# Run the random agent example
python examples/random_nim.py

# Run the Q-table learning example
python examples/qtable.py
```

## Architecture and Key Components

### Environment Structure
- **NimEnv** (`gym_nim/envs/nim_env.py`): Main environment class implementing the Nim game
  - Uses discrete action and observation spaces
  - Action space: tuple of (pile_index, count_to_remove)
  - State contains board configuration and current player
  - Supports move generation for legal moves
  - Implements reward structure: -1 for losing, -2 for illegal moves

### Key Implementation Details
- The environment uses the legacy Gym API with `_step()`, `_reset()`, and `_render()` methods (underscore prefixed)
- Board is initialized with 3 piles containing [7, 5, 3] pieces by default
- Players alternate between 1 and 2 using `3 - current_player` pattern
- Move legality checking is implemented but currently hardcoded for 3 piles
- The environment tracks whose turn it is in the state dictionary

### Testing Considerations
- Comprehensive pytest test suite with 99% code coverage
- The project includes GitHub Actions CI configuration testing Python 3.9-3.12
- Docker testing uses Python 3.9 with pytest for consistent test environment

### Development Notes
- Many TODO comments indicate areas for generalization (variable number of piles, different starting configurations)
- The action space definition (Discrete(9)) doesn't match the actual action format (tuple)
- Examples demonstrate both random play and Q-learning approaches
- Reward negation is handled in the example code for player 2 perspective

## Migration to Gymnasium

### Key Considerations for Gymnasium Migration
Based on successful migration patterns from similar projects:

1. **API Changes**
   - Update `_step()`, `_reset()`, `_render()` to `step()`, `reset()`, `render()` (remove underscores)
   - `reset()` should return `(observation, info)` tuple instead of just observation
   - `step()` returns `(observation, reward, terminated, truncated, info)` instead of `(observation, reward, done, info)`
   - Update `render()` to handle new rendering modes

2. **Package Structure Updates**
   - Environment registration remains in `gym_nim/__init__.py` but uses `gymnasium.register()`
   - Update inheritance from `gym.Env` to `gymnasium.Env`
   - Consider creating `gym_nim/_version.py` for centralized version management

3. **Testing Strategy**
   - Test suite already implemented with comprehensive coverage
   - All environment mechanics and edge cases are tested
   - Integration tests for examples are included
   - CI matrix testing across Python 3.9-3.12 is configured

4. **Development Workflow for Migration**
   - Create feature branch: `feature/gymnasium-migration`
   - Update dependencies in `setup.py` from `gym==0.9.1` to latest `gymnasium`
   - Run tests frequently during migration to catch API incompatibilities
   - Update Docker configuration to test with new dependencies

5. **Version Management**
   - Consider implementing centralized version management
   - Create update script to sync versions across setup.py and requirements files
   - Document supported Python version range (recommend 3.9-3.12)

## Claude Code Workflow Hints

- Prefer writing minimal, focused code changes
- Always run tests after making modifications
- Use feature branches for substantial refactoring
- Maintain clear, descriptive commit messages
- Document significant architectural decisions