import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from Gridworld import Gridworld

# ==========================================
# 參數調整區
# ==========================================
GAME_MODE = 'random'  # 可選 'static', 'player', 'random'
EPSILON = 0.1         # ϵ-greedy 探索率
GAMMA = 0.9           # 折扣因子 (γ)
EPISODES = 500        # 訓練回合數
LR = 0.001            # 學習率
# ==========================================

class NaiveDQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(NaiveDQN, self).__init__()
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

# 環境初始化
env = Gridworld(size=4, mode=GAME_MODE)
input_shape = env.board.render_np().shape 
action_map = ['u', 'd', 'l', 'r']
model = NaiveDQN(input_shape, len(action_map))
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

reward_history = []

for ep in range(EPISODES):
    # 重置環境
    if GAME_MODE == 'static': env.initGridStatic()
    elif GAME_MODE == 'player': env.initGridPlayer()
    else: env.initGridRand()
    
    state = torch.from_numpy(env.board.render_np()).float().unsqueeze(0)
    total_reward = 0
    done = False
    step_count = 0
    
    # 注意：這裡只有抵達終點才會結束，或是步數超過 30 
    while not done and step_count < 30:
        if np.random.rand() < EPSILON:
            action_idx = np.random.randint(len(action_map))
        else:
            with torch.no_grad():
                q_values = model(state)
                action_idx = torch.argmax(q_values).item()
        
        action = action_map[action_idx]
        env.makeMove(action)
        reward = env.reward()
        new_state = torch.from_numpy(env.board.render_np()).float().unsqueeze(0)
        
        # 修改後的判定邏輯：
        # 只有吃到目標 (+) 才結束遊戲
        if reward == 20:
            done = True
        
        # 即使踩到陷阱 (-50)，我們也只是計算更新，但不設 done = True
        with torch.no_grad():
            next_q_values = model(new_state)
            max_next_q = torch.max(next_q_values).item()
        
        # 如果是目標，則沒有下一步 Q 值；如果是陷阱，則繼續計算未來獎勵
        target_q = reward + (GAMMA * max_next_q if not done else 0)
        
        current_q = model(state)[0, action_idx]
        loss = loss_fn(current_q, torch.tensor(target_q, dtype=torch.float))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = new_state
        total_reward += reward
        step_count += 1
        
    reward_history.append(total_reward)
    if (ep + 1) % 50 == 0:
        avg_last_50 = np.mean(reward_history[-50:])
        print(f"Episode {ep+1}/{EPISODES}, Avg Reward (last 50): {avg_last_50:.2f}")

# ==========================================
# 繪圖區塊 (固定座標軸與版面)
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
plt.title(f"DQN Training Performance (Mode: {GAME_MODE})")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()