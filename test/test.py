import numpy as np
import random

class MinimaxQAgent:
    def __init__(self, env, player, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.player = player
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {}
        
    def policy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            actions = self.env.get_available_actions(state)
            values = np.array([self.Q.get((state, a), 0) for a in actions])
            max_q = np.max(values)
            indices = np.where(values == max_q)[0]
            return actions[np.random.choice(indices)]
        
    def learn(self, state, action, next_state, reward, done):
        next_actions = self.env.get_available_actions(next_state)
        if done:
            next_Q_value = 0
        else:
            next_Q_values = [self.Q.get((next_state, a), 0) for a in next_actions]
            if self.player == 'X':
                next_Q_value = np.min(next_Q_values)
            else:
                next_Q_value = np.max(next_Q_values)
        q_key = (state, action)
        if q_key not in self.Q:
            self.Q[q_key] = 0
        if done:
            self.Q[q_key] += self.alpha * (reward - self.Q[q_key])
        else:
            self.Q[q_key] += self.alpha * (reward + self.gamma * next_Q_value - self.Q[q_key])
            
    def train(self, num_episodes=1000):
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action, self.player)
                self.learn(state, action, next_state, reward, done)
                state = next_state
