import gym
import numpy as np
import queue
import gym_tic_tac_toe
import random
# based on https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# "Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks" by Arthur Juliani


def pick_min(choices, moves):
    minc = +99999999
    for i in range(len(moves)):
        m = moves[i]
        c = choices[0][m[1]]
        if (c < minc):
            best_move = m
            minc = c
    
    return best_move

def pick_max(choices, moves):
    maxc = -99999999
    for i in range(len(moves)):
        m = moves[i]
        c = choices[0][m[1]]
    #               print ("choice: ", c)
        if (c > maxc):
            best_move = m
            maxc = c
    
    return best_move

def hash_ttt(state):
    #of course this is just for the upper bound;
    #we should really take advantage of the redundancies
    # to reduce the number of states to 765 for the board
    # and who is on move really is implicit in how many
    # squares are occupied
    
    #print("hashing state: ", state)
    retval = 0
    low9 = 0
    high9 = 0
    lowmult = 2
    highmult = 1024
    board = state.get('board')
    if (state['on_move'] == -1):
        retval = 1
    for i in range(9):
        if (board[i] != 0):
            retval |= lowmult
            if (board[i] < 0):
                retval |= highmult
        lowmult *=2
        highmult *= 2
    return retval
def random_plus_middle_move(moves, p):
    if ([p, 4] in moves):
        m = [p, 4]
    else:
        m = random_move(moves, p)
    return m
def random_move(moves, p):
    m = random.choice(moves)
    return m


env = gym.make('tic_tac_toe-v0')
asp = env.action_space
print (asp)

os = env.observation_space
print (os)


#Initialize table with all zeros
Q = np.zeros([os.n,asp.n])
# Set learning parameters
lr = .85
y = .99
num_episodes = 10000
ROLLING_ELEMENTS = 1000.0
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
q = queue.Queue(ROLLING_ELEMENTS)
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #print (Q)
        #Choose an action by greedily (with noise) picking from Q table
        hs = hash_ttt(s)
#         print ("s: ", s)
#         print ("hs: ", hs)
        pick =  np.random.randn(1,asp.n)
        puck = (1./(i+1))
        bonus = pick * puck
 #       print ("pick: ", pick, ", puck: ", puck, ", bonus: ", bonus)
        choices = Q[hs,:] +bonus
#        print ("choices: " , choices)
        om = s['on_move']
        moves = env.move_generator()
#        print ("legal moves: ", moves)
        a = []
        if len(moves) == 0:
            break
        elif len(moves) == 1:
            a = moves[0]
        if (om == 1):
            a = pick_max(choices, moves)
            
#            a = [om, np.argmax(choices)]
        else: # om == -1
#            a = [om, np.argmin(choices)]
            a= pick_min(choices, moves)
            
            #a = random_move(moves, om)
          
        #Get new state and reward from environment
#        print('action: ', a)
        s1,r,d,_ = env.step(a)
        hs1 = hash_ttt(s1)
#        print ('new state: ', s1, " (", hs1, "), reward: ", r)
        rAll += r
        #Update Q-Table with new knowledge
        if (om == 1):
            Q[hs,a] = Q[hs,a] + lr*(r + y*np.max(Q[hs1,:]) - Q[hs,a])
        else:
            Q[hs,a] = Q[hs,a] + lr*(r + y*np.min(Q[hs1,:]) - Q[hs,a])
          
        s = s1
        if (d == True):
          break
    sz = q.qsize()
    if (q.full()):
        q.get()#remove one item to make room
    q.put(rAll)
    rolling_rAll = 0
    if(q.full()):
        for k in range(int(ROLLING_ELEMENTS)):
            it = q.get()
            #print (it)
            rolling_rAll += it 
            q.put(it)
        if (i % 500 == 0):
            print (rolling_rAll/ROLLING_ELEMENTS)
    rList.append(rAll)
#     if (rolling_rAll >= 9):
#       print ("good at ", i)
      #break

print ("Score over time: " +  str(sum(rList)/num_episodes))
print ("Final Q-Table Values")
print (Q)
