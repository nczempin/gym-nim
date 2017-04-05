# gym-nim

An example of a custom environment for https://github.com/openai/gym.

I want to try out self-play in a Reinforcement Learning context. Rather than the board game environments on openai/gym right now, which are "single-player" by providing a built-in opponent, I want to create an agent that learns a strategy by playing against itself, so it will try to maximize the reward for "player 1" and minimize it for "player 2".

A game that's even simpler than Tic-Tac-Toe/Noughts & Crosses is Nim.
