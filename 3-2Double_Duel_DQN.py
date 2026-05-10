import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from Gridworld import Gridworld

# ==========================================
# 參數調整區
# ==========================================
GAME_MODE = 'random'
EPSILON = 0.1
GAMMA = 0.9
EPISODES = 500
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 2000
SYNC_TARGET_STEPS = 50  # 同步步數改為 50
# ==========================================

# Dueling DQN 網路架構
class DuelingDQNNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DuelingDQNNet, self).__init__()
        self.flatten = nn.Flatten()
        
        # 共同特徵提取
        self.feature_layer = nn.Sequential(
            nn.Linear(np.prod(input_shape), 64),
            nn.ReLU()
        )
        
        # 狀態價值支流 (V)
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 動作優勢支流 (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 結合 V 與 A 得到 Q 值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# 初始化環境
env = Gridworld(size=4, mode=GAME_MODE)
input_shape = env.board.render_np().shape
action_map = ['u', 'd', 'l', 'r']
n_actions = len(action_map)

# 初始化雙網路
policy_net = DuelingDQNNet(input_shape, n_actions)
target_net = DuelingDQNNet(input_shape, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

memory = deque(maxlen=MEMORY_SIZE)
reward_history = []
total_steps = 0 

for ep in range(EPISODES):
    if GAME_MODE == 'static': env.initGridStatic()
    elif GAME_MODE == 'player': env.initGridPlayer()
    else: env.initGridRand()
    
    state = env.board.render_np()
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < 30:
        # ϵ-greedy 探索
        if np.random.rand() < EPSILON:
            action_idx = np.random.randint(n_actions)
        else:
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_idx = torch.argmax(policy_net(state_t)).item()
        
        action = action_map[action_idx]
        env.makeMove(action)
        reward = env.reward()
        next_state = env.board.render_np()
        
        if reward == 20: 
            done = True
            
        memory.append((state, action_idx, reward, next_state, done))
        
        if len(memory) > BATCH_SIZE:
            minibatch = random.sample(memory, BATCH_SIZE)
            
            states_b = torch.tensor([m[0] for m in minibatch], dtype=torch.float)
            actions_b = torch.tensor([m[1] for m in minibatch], dtype=torch.long).unsqueeze(1)
            rewards_b = torch.tensor([m[2] for m in minibatch], dtype=torch.float)
            next_states_b = torch.tensor([m[3] for m in minibatch], dtype=torch.float)
            dones_b = torch.tensor([m[4] for m in minibatch], dtype=torch.float)
            
            # 當前 Q 值
            current_q = policy_net(states_b).gather(1, actions_b).squeeze()
            
            # --- Double DQN 核心更新邏輯 ---
            with torch.no_grad():
                # 1. 由 Policy Net 選出下一狀態的最佳動作
                best_actions = torch.argmax(policy_net(next_states_b), dim=1).unsqueeze(1)
                # 2. 由 Target Net 評估該動作的 Q 值
                next_q_values = target_net(next_states_b).gather(1, best_actions).squeeze()
                # 3. 計算目標值
                target_q = rewards_b + (GAMMA * next_q_values * (1 - dones_b))
            
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每 50 步更新一次 Target Network
            total_steps += 1
            if total_steps % SYNC_TARGET_STEPS == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
        state = next_state
        total_reward += reward
        step_count += 1
        
    reward_history.append(total_reward)
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{EPISODES}, Total Reward: {total_reward}")

# ==========================================
# 繪圖區塊
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(reward_history, alpha=0.3, color='blue', label='Raw Reward')

window = 10
if len(reward_history) >= window:
    moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(reward_history)), moving_avg, 
             color='orange', linewidth=2, label=f'{window}-Ep Moving Avg')

if GAME_MODE == 'random':
    plt.ylim(-300, 50)
else:
    plt.ylim(-100, 50)

plt.title(f"Double Dueling DQN with Replay (Mode: {GAME_MODE})")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()