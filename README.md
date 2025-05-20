# gym-nim

An example of a custom environment for https://github.com/openai/gym.

I want to try out self-play in a Reinforcement Learning context. Rather than the board game environments on openai/gym right now, which are "single-player" by providing a built-in opponent, I want to create an agent that learns a strategy by playing against itself, so it will try to maximize the reward for "player 1" and minimize it for "player 2".

A game that's even simpler than Tic-Tac-Toe/Noughts & Crosses is Nim.

## Docker-based build pipeline

A `Dockerfile` is provided to test the project with the legacy `gym==0.9.1` dependency. Build and run the container to execute the test suite:

```bash
docker build -t gym-nim .
docker run --rm gym-nim
```

This runs `tests.py` inside the container so you can verify that the package still functions with the old dependency.
