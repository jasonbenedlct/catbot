import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

def decode_state(state):
    bot_r = state // 1000
    bot_c = (state // 100) % 10
    cat_r = (state // 10) % 10
    cat_c = state % 10
    return (bot_r, bot_c), (cat_r, cat_c)

def manhattan_distance(pos1, pos2):
    """Calculates Manhattan distance between two grid cells."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    alpha = 0.15                 # Learning rate
    gamma = 0.95                 # Discount factor
    epsilon = 1.0                # Exploration rate
    epsilon_min = 0.01           # Minimum exploration rate
    epsilon_decay = 0.999        # Decay rate per episode
    max_steps_per_episode = 100  # Cap steps to prevent infinite training loops
    num_actions = env.action_space.n

    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
               
        
        state, info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            steps += 1
            
            # 1. Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = int(np.argmax(q_table[state]))  # Exploit

            # 2. Take action in environment
            next_state, _, done, truncated, _ = env.step(action)

            # 3. Decode positions for custom reward calculations
            bot_pos, cat_pos = decode_state(state)
            next_bot_pos, next_cat_pos = decode_state(next_state)

            prev_dist = manhattan_distance(bot_pos, cat_pos)
            new_dist = manhattan_distance(next_bot_pos, next_cat_pos)

            # 4. Compute reward manually
            if done:
                reward = 100.0  # Successfully caught the cat[cite: 1]
            else:
                # Optimal policy convergence without penalizing for cat's evasion
                shaping = gamma * (-new_dist) - (-prev_dist)
                reward = shaping - 0.1

            # 5. Update Q-table (Bellman Equation)
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward if done else reward + gamma * q_table[next_state][best_next_action]
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            if done or truncated:
                break

        # Decay exploration rate over time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table
