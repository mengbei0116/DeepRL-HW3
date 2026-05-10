import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import pytorch_lightning as pl
from Gridworld import Gridworld

# ==========================================
# 參數調整區 (維持原始變因)
# ==========================================
GAME_MODE = 'random'   
EPSILON = 0.1
GAMMA = 0.9
EPISODES = 3000
LR = 0.001
BATCH_SIZE = 64        
MEMORY_SIZE = 2000     
SYNC_TARGET_STEPS = 100 
MAX_STEPS = 50         

# ==========================================
# Dueling DQN 網路架構
# ==========================================
class DuelingDQNNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(np.prod(input_shape), 256),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# ==========================================
# PyTorch Lightning 訓練模組
# ==========================================
class DQNLitModule(pl.LightningModule):
    def __init__(self, env, learning_rate=1e-3):
        super().__init__()
        self.env = env
        input_shape = env.board.render_np().shape
        self.n_actions = 4
        self.action_map = ['u', 'd', 'l', 'r']
        
        self.policy_net = DuelingDQNNet(input_shape, self.n_actions)
        self.target_net = DuelingDQNNet(input_shape, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.buffer = deque(maxlen=MEMORY_SIZE)
        self.reward_history = []
        self.total_steps = 0
        self.lr = learning_rate
        
        # 核心：設置手動優化
        self.automatic_optimization = False

    def select_action(self, state):
        if np.random.rand() < EPSILON:
            return np.random.randint(self.n_actions)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state_t)).item()

    def training_step(self, batch, batch_idx):
        if GAME_MODE == 'static': self.env.initGridStatic()
        elif GAME_MODE == 'player': self.env.initGridPlayer()
        else: self.env.initGridRand()
        
        state = self.env.board.render_np()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < MAX_STEPS:
            action_idx = self.select_action(state)
            self.env.makeMove(self.action_map[action_idx])
            reward = self.env.reward()
            next_state = self.env.board.render_np()
            done = True if reward == 20 else False
            
            self.buffer.append((state, action_idx, reward, next_state, done))
            
            if len(self.buffer) > BATCH_SIZE:
                opt = self.optimizers()
                minibatch = random.sample(self.buffer, BATCH_SIZE)
                
                sb = torch.tensor([m[0] for m in minibatch], dtype=torch.float).to(self.device)
                ab = torch.tensor([m[1] for m in minibatch], dtype=torch.long).unsqueeze(1).to(self.device)
                rb = torch.tensor([m[2] for m in minibatch], dtype=torch.float).to(self.device)
                nb = torch.tensor([m[3] for m in minibatch], dtype=torch.float).to(self.device)
                db = torch.tensor([m[4] for m in minibatch], dtype=torch.float).to(self.device)

                # Double DQN 邏輯
                curr_q = self.policy_net(sb).gather(1, ab).squeeze()
                with torch.no_grad():
                    next_actions = torch.argmax(self.policy_net(nb), dim=1).unsqueeze(1)
                    next_q = self.target_net(nb).gather(1, next_actions).squeeze()
                    target_q = rb + (GAMMA * next_q * (1 - db))

                loss = nn.MSELoss()(curr_q, target_q)
                
                opt.zero_grad()
                self.manual_backward(loss)
                
                # 在手動優化模式下，必須在此手動進行梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                
                opt.step()

                self.total_steps += 1
                if self.total_steps % SYNC_TARGET_STEPS == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            state = next_state
            total_reward += reward
            step_count += 1
        
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
            
        self.reward_history.append(total_reward)
        return torch.tensor(total_reward)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPISODES)
        return [optimizer], [scheduler]

# ==========================================
# 啟動訓練與繪圖
# ==========================================
if __name__ == "__main__":
    grid_env = Gridworld(size=4, mode=GAME_MODE)
    model = DQNLitModule(grid_env, learning_rate=LR)
    
    # 修正 MisconfigurationException：移除 gradient_clip_val
    trainer = pl.Trainer(
        max_epochs=EPISODES,
        enable_checkpointing=False,
        logger=False,
        devices=1 if torch.cuda.is_available() else 0
    )

    trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader([0]))

    # --- 繪圖 ---
    reward_history = model.reward_history
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, alpha=0.3, color='blue', label='Raw Reward')

    window = 10
    if len(reward_history) >= window:
        moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(reward_history)), moving_avg, 
                 color='orange', linewidth=2, label=f'{window}-Ep Moving Avg')

    plt.ylim((-300 if GAME_MODE == 'random' else -100), 50)
    plt.title(f"Double Dueling DQN (Lightning)\nMode: {GAME_MODE}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()