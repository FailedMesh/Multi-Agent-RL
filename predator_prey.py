import numpy as np
import gym

def state_to_encoding(state):
    encoding = 0
    for i in range(len(state)):
        encoding += (5**(5-i))*int(state[i])
    return encoding

def encoding_to_state(encoding):
    state = ""
    for i in range(5, 0, -1):
        remainder = encoding % 5
        encoding = encoding//5
        state = str(remainder) + state
    remainder = encoding % 5
    state = str(remainder) + state
    return state

def action_to_encoding(action):
    encoding = 0
    for i in range(len(action)):
        encoding += (5**(1-i))*int(action[i])
    return encoding

def encoding_to_action(encoding):
    action = ""
    remainder = encoding % 5
    encoding = encoding//5
    action = str(int(remainder)) + action
    remainder = encoding % 5
    action = str(int(remainder)) + action
    return action

def get_next_state(state, action):
    
    #Encoding in general means the number, state is the string
    state = encoding_to_state(state)
    action = encoding_to_action(action)
    
    noop_1 = 0
    noop_2 = 0
    
    #STATE:
    #First 2, Second 2, Third 2, depict the positions of the 1st pred, 2nd pred, and prey respectively.
    
    #PREDATOR 1:
    
    if action[0] == "0":
        if state[0] != "4":
            state = state[:0] + str(int(state[0]) + 1) + state[1:]
    if action[0] == "1":
        if state[1] != "0":
            state = state[:1] + str(int(state[1]) - 1) + state[2:]
    if action[0] == "2":
        if state[0] != "0":
            state = state[:0] + str(int(state[0]) - 1) + state[1:]
    if action[0] == "3":
        if state[1] != "4":
            state = state[:1] + str(int(state[1]) + 1) + state[2:]
    if action[0] == "4":
        if (abs(int(state[4]) - int(state[0])) == 1 and abs(int(state[5]) - int(state[1])) == 0) or (abs(int(state[4]) - int(state[0])) == 0 and abs(int(state[5]) - int(state[1])) == 1):
            noop_1 = 1
        
    #PREDATOR 2:
    
    if action[1] == "0":
        if state[2] != "4":
            state = state[:2] + str(int(state[2]) + 1) + state[3:]
    if action[1] == "1":
        if state[3] != "0":
            state = state[:3] + str(int(state[3]) - 1) + state[4:]
    if action[1] == "2":
        if state[2] != "0":
            state = state[:2] + str(int(state[2]) - 1) + state[3:]
    if action[1] == "3":
        if state[3] != "4":
            state = state[:3] + str(int(state[3]) + 1) + state[4:]
    if action[1] == "4":
        if (abs(int(state[4]) - int(state[2])) == 1 and abs(int(state[5]) - int(state[3])) == 0) or (abs(int(state[4]) - int(state[2])) == 0 and abs(int(state[5]) - int(state[3])) == 1):
            noop_2 = 1
            
    reward_array = np.zeros((4,))
    state_array = [state, state, state, state]
    prev_state_array = state_array[:]
    
    #All Possible Next States:
    
    if state_array[0][4] != "4":
        state_array[0] = state_array[0][:4] + str(int(state_array[0][4]) + 1) + state_array[0][5:]
    if state_array[1][5] != "0":
        state_array[1] = state_array[1][:5] + str(int(state_array[1][5]) - 1)
    if state_array[2][4] != "0":
        state_array[2] = state_array[2][:4] + str(int(state_array[2][4]) - 1) + state_array[2][5:]
    if state_array[3][5] != "4":
        state_array[3] = state_array[3][:5] + str(int(state_array[3][5]) + 1)
    
            
    #TERMINAL CASES:
    
    for i in range(len(state_array)):

#         if action[0] == "4":
#             if abs(int(state_array[i][4]) - int(state_array[i][0])) == 1 or abs(int(state_array[i][5]) - int(state_array[i][1])) == 1:
#                 noop_1 = 1

#         if action[1] == "4":
#             if abs(int(state_array[i][4]) - int(state_array[i][2])) == 1 or abs(int(state_array[i][5]) - int(state_array[i][3])) == 1:
#                 noop_2 = 1
        
        if (noop_1 + noop_2) == 2:          #Correct Noop
            state_array[i] = state_to_encoding(state_array[i])
            reward_array[i] = 1
        elif (noop_1 + noop_2) == 1:        #Wrong Noop       
            state_array[i] = state_to_encoding(state_array[i])
            reward_array[i] = -0.5
        elif (action[0] == "4" and noop_1 == 0) or (action[1] == "4" and noop_2 == 0):  #Terrible Noop
            state_array[i] = state_to_encoding(state_array[i])
            reward_array[i] = -1
        #Collision:
        elif (state_array[i][:2] == state_array[i][2:4]) or (state_array[i][2:4] == state_array[i][4:]) or (state_array[i][:2] == state_array[i][4:]):
            state_array[i] = state_to_encoding(prev_state_array[:][i])
            reward_array[i] = -0.5
        #Movement Collision:
        elif (state_array[i][:2] == prev_state_array[i][2:4] and prev_state_array[i][:2] == state_array[i][2:4]) or (state_array[i][4:] == prev_state_array[i][2:4] and prev_state_array[i][4:] == state_array[i][2:4]) or (state_array[i][:2] == prev_state_array[i][4:] and prev_state_array[i][:2] == state_array[i][4:]):
            state_array[i] = state_to_encoding(prev_state_array[:][i])
            reward_array[i] = -0.5
        else:
            state_array[i] = state_to_encoding(state_array[i])
            reward_array[i] = -0.01
        
    
    return reward_array, state_array

def get_custom_reward(state, action, next_state):
    
    
    #Encoding in general means the number, state is the string
    state = encoding_to_state(state)
    action = encoding_to_action(action)
    next_state = encoding_to_state(next_state)
    
    noop_1 = 0
    noop_2 = 0
    
    #PREDATOR 1 NOOPING:
    if (abs(int(next_state[4]) - int(next_state[0])) == 1 and abs(int(next_state[5]) - int(next_state[1])) == 0) or (abs(int(next_state[4]) - int(next_state[0])) == 0 and abs(int(next_state[5]) - int(next_state[1])) == 1):
        noop_1 = 1
    
    #PREDATOR 2 NOOPING:
    if (abs(int(next_state[4]) - int(next_state[2])) == 1 and abs(int(next_state[5]) - int(next_state[3])) == 0) or (abs(int(next_state[4]) - int(next_state[2])) == 0 and abs(int(next_state[5]) - int(next_state[3])) == 1):
        noop_2 = 1

    #REWARD ASSIGNMENT:
    
    if (noop_1 + noop_2) == 2:          #Correct Noop
        reward = 1
        
#     elif (noop_1 + noop_2) == 1:        #Wrong Noop       
#         reward = -0.5
    
    elif (action[0] == "4" and noop_1 == 0) or (action[1] == "4" and noop_2 == 0):  #Terrible Noop
        reward = -1
   
    #Collision:
    elif (next_state[:2] == next_state[2:4]) or (next_state[2:4] == next_state[4:]) or (next_state[:2] == next_state[4:]):
        reward = -0.5
    
    #Movement Collision:
    elif (next_state[:2] == state[2:4] and state[:2] == next_state[2:4]) or (next_state[4:] == state[2:4] and state[4:] == next_state[2:4]) or (next_state[:2] == state[4:] and state[:2] == next_state[4:]):
        reward = -0.5
    
    else:
        reward = -0.01
        
    
    return reward