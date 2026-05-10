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
BATCH_SIZE = 32        # 每次訓練抽取的樣本數
MEMORY_SIZE = 2000     # Buffer 最大容量
# ==========================================

class DQNNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQNNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_shape), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# 初始化環境與 Buffer
env = Gridworld(size=4, mode=GAME_MODE)
input_shape = env.board.render_np().shape
action_map = ['u', 'd', 'l', 'r']
n_actions = len(action_map)

model = DQNNet(input_shape, n_actions)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# 經驗回放池
memory = deque(maxlen=MEMORY_SIZE)

reward_history = []

for ep in range(EPISODES):
    if GAME_MODE == 'static': env.initGridStatic()
    elif GAME_MODE == 'player': env.initGridPlayer()
    else: env.initGridRand()
    
    state = env.board.render_np()
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < 30:
        # ϵ-greedy 選擇動作
        if np.random.rand() < EPSILON:
            action_idx = np.random.randint(n_actions)
        else:
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_idx = torch.argmax(model(state_t)).item()
        
        action = action_map[action_idx]
        env.makeMove(action)
        reward = env.reward()
        next_state = env.board.render_np()
        
        # 只有吃到終點才結束
        if reward == 20:
            done = True
            
        # 存入 Buffer
        memory.append((state, action_idx, reward, next_state, done))
        
        # 當 Buffer 累積足夠資料時開始訓練
        if len(memory) > BATCH_SIZE:
            # 隨機抽樣一個 Batch
            minibatch = random.sample(memory, BATCH_SIZE)
            
            # 將資料轉換為 Tensor (優化效能一次處理整個 Batch)
            states_b = torch.tensor([m[0] for m in minibatch], dtype=torch.float)
            actions_b = torch.tensor([m[1] for m in minibatch], dtype=torch.long).unsqueeze(1)
            rewards_b = torch.tensor([m[2] for m in minibatch], dtype=torch.float)
            next_states_b = torch.tensor([m[3] for m in minibatch], dtype=torch.float)
            dones_b = torch.tensor([m[4] for m in minibatch], dtype=torch.float)
            
            # 計算當前 Q 值
            current_q = model(states_b).gather(1, actions_b).squeeze()
            
            # 計算 Target Q 值
            with torch.no_grad():
                max_next_q = torch.max(model(next_states_b), dim=1)[0]
                target_q = rewards_b + (GAMMA * max_next_q * (1 - dones_b))
            
            # 更新網路
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        state = next_state
        total_reward += reward
        step_count += 1
        
    reward_history.append(total_reward)
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{EPISODES}, Total Reward: {total_reward}")

# ==========================================
# 繪圖區塊 
# ==========================================
plt.figure(figsize=(10, 6))  # 固定版面大小，方便不同實驗對比

# 1. 繪製原始數據 (淺色)
plt.plot(reward_history, alpha=0.3, color='blue', label='Raw Reward')

# 2. 計算並繪製每 10 回合平均曲線
window = 10
if len(reward_history) >= window:
    moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    # 使用 range 確保 X 軸對齊 (從第 9 個 index 開始，即第 10 回合)
    plt.plot(range(window - 1, len(reward_history)), moving_avg, 
             color='orange', linewidth=2, label=f'{window}-Ep Moving Avg')

# 3. 根據遊戲模式設定特定的 Y 軸範圍
if GAME_MODE == 'random':
    plt.ylim(-300, 50)  # Random 模式：+50 ~ -300
else:
    plt.ylim(-100, 50)  # Static 與 Player 模式：+50 ~ -100

# 4. 圖表裝飾
plt.title(f"DQN with Experience Replay (Mode: {GAME_MODE})")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()