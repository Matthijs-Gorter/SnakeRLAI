import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import time
import pickle
import os


MAX_MEMORY = 1_000_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.model = Linear_QNet(11, 256, 3)
        self.target_model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        
        self.update_target_network()
        self.load_model_and_memory()

def save_model_and_memory(self):
    self.model.save()
    with open('./model/memory.pkl', 'wb') as f:
        pickle.dump(self.memory, f)
    with open('./model/n_games.pkl', 'wb') as f:
        pickle.dump(self.n_games, f)

def load_model_and_memory(self):
    if os.path.exists('./model/model.pth'):
        self.model.load('./model/model.pth')
    if os.path.exists('./model/memory.pkl'):
        with open('./model/memory.pkl', 'rb') as f:
            self.memory = pickle.load(f)
    if os.path.exists('./model/n_games.pkl'):
        with open('./model/n_games.pkl', 'rb') as f:
            self.n_games = pickle.load(f)


    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    start_time = time.time()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    visualise = False
    last_10_scores = deque(maxlen=10)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move, visualise)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            last_10_scores.append(score)
            if len(last_10_scores) == 10:
                mean_last_10_scores = sum(last_10_scores) / 10
            else:
                mean_last_10_scores = sum(last_10_scores) / len(last_10_scores)
            plot_mean_scores.append(mean_last_10_scores)

            if agent.n_games % 10 == 0:
                agent.update_target_network()
                agent.save_model_and_memory()

            
                end_time = time.time()
                print('Game:', agent.n_games, '\tScore:', score, '\tRecord:', record, "\tElapsed time (s):", end_time - start_time)


            if agent.n_games >= 200:
                plot(plot_scores, plot_mean_scores)
                visualise = True


if __name__ == '__main__':
    train()