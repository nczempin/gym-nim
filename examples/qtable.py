# based on https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# "Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks" by Arthur Juliani


import gym
import numpy as np
import queue
import gym_nim
import random

def hash_nim_move(move):
    a = move[0]
    b = move[1]
    return 3 * a + b -1

def pick_min(choices, moves):
    minc = +99999999
    for i in range(len(moves)):
        m = moves[i]
        c = choices[0][hash_nim_move(m)]
        if (c < minc):
            best_move = m
            minc = c
    
    return best_move

def pick_max(choices, moves):
    maxc = -99999999
    for i in range(len(moves)):
        m = moves[i]
        c = choices[0][m[0]]
    #               print ("choice: ", c)
        if (c > maxc):
            best_move = m
            maxc = c
    
    return best_move

def hash_nim_state(state):
    # of course this is just for the upper bound;
    # we should really take advantage of the redundancies
    # to reduce the number of states to 765 for the board
    # and who is on move really is implicit in how many
    # squares are occupied
    
    # print("hashing state: ", state)
    retval = 0
    
    #TODO: sort the values to take advantage of equivalence
    

    board = state.get('board')
    if (state['on_move'] == 2):
        retval = 1
        
    retval += (board[0]) << 1
    retval += ((board[1]) << (1 + 3))
    retval += ((board[2]) << (1 + 3 + 3))
    if (retval > 1023):
        print("invalid: ", retval)
        print (board)
    return retval

def random_move(moves):
    m = random.choice(moves)
    return m
def choose_move(Q, hs, env, om, maximizing_player, bonus):
    choices = Q[hs, :] + bonus
#        print ("choices: " , choices)
    moves = env.move_generator()
#        print ("legal moves: ", moves)
    a = []
    if len(moves) == 1:
        a = moves[0]
    elif len(moves) > 0:
        if (om == maximizing_player):
            a = pick_max(choices, moves)
        
#            a = [om, np.argmax(choices)]
        else:  # om == -1
#            a = [om, np.argmin(choices)]
            #a = pick_min(choices, moves)
            a = random_move(moves)
            
    return a

def train(env):

    action_space = env.action_space
    observation_space = env.observation_space
    
    
    # Initialize table with all zeros
    Q = np.zeros([observation_space.n, action_space.n])
    # Set learning parameters
    lr = .85  # learning rate
    y = .99  # discount factor.
    num_episodes = 10000
    ROLLING_ELEMENTS = 1000.0
    # create lists to contain total rewards and steps per episode
    # jList = []
    rList = []
    q = queue.Queue(ROLLING_ELEMENTS)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        env.set_board([2,1,1])
        maximizing_player = 1  # TODO randomize; for that we need to generalize the reward function
        rAll = 0
        d = False
        j = 0
        # The Q-Table learning algorithm
        while j < 99:
            j += 1
            # print (Q)
            # Choose an action by greedily (with noise) picking from Q table
      #         print ("s: ", s)
    #         print ("hs: ", hs)
            pick = np.random.randn(1, action_space.n)
            puck = (1. / (i + 1))
            bonus = pick * puck
     #       print ("pick: ", pick, ", puck: ", puck, ", bonus: ", bonus)
            om = s['on_move']
            hs = hash_nim_state(s)
            a = choose_move(Q, hs, env, om, maximizing_player, bonus)
            if (not a):
                break
                # a = random_move(moves, om)
              
            # Get new state and reward from environment
    #        print('action: ', a)
            s1, reward, d, _ = env.step(a)
            if (om == 2):
                reward = - reward
            hs1 = hash_nim_state(s1)
    #        print ('new state: ', s1, " (", hs1, "), reward: ", r)
            rAll += reward
            # Update Q-Table with new knowledge
            if (om == maximizing_player):
                Q[hs, a] = Q[hs, a] + lr * (reward + y * np.max(Q[hs1, :]) - Q[hs, a])
            else:
                Q[hs, a] = Q[hs, a] + lr * (reward + y * np.min(Q[hs1, :]) - Q[hs, a])
              
            s = s1
            if (d == True):
              break
        sz = q.qsize()
        if (q.full()):
            q.get()  # remove one item to make room
        q.put(rAll)
        rolling_rAll = 0
        if(q.full()):
            for k in range(int(ROLLING_ELEMENTS)):
                it = q.get()
                # print (it)
                rolling_rAll += it 
                q.put(it)
            if (i % 500 == 0):
                print (rolling_rAll / ROLLING_ELEMENTS)
                for i in range(observation_space.n):
                    for j in range (action_space.n):
                        q2 = Q[i][j]
                        if (q2 != 0):
                            print (i, ", ", j, ": ",q2)
        rList.append(rAll)
    #     if (rolling_rAll >= 9):
    #       print ("good at ", i)
          # break
    
    print ("Score over time: " + str(sum(rList) / num_episodes))
    print ("Final Q-Table Values")
    return Q

env = gym.make('nim-v0')

Q = train(env)

print (Q)
# Access spaces directly from env since os and asp are local to train()
observation_space = env.observation_space
action_space = env.action_space
for i in range(observation_space.n):
    for j in range(action_space.n):
        q = Q[i][j]
        if (q != 0):
            print (i, ", ", j, ": ",q)

s = env.reset()
