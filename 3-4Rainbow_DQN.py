import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math
from collections import deque
import pytorch_lightning as pl
from Gridworld import Gridworld

# 效能優化
torch.set_float32_matmul_precision('high')

# ==========================================
# 參數調整區 (優化後的 Rainbow 配置)
# ==========================================
GAME_MODE = 'random'
GAMMA = 0.95           # 降低 Gamma，讓 Agent 更關注近期獎勵
N_STEPS = 3            
V_MIN, V_MAX = -20, 30 # 縮小 C51 範圍以提高解析度
N_ATOMS = 51           
EPISODES = 3000
LR = 0.0005            # 稍微提高學習率配合 Scheduler
BATCH_SIZE = 64        
MEMORY_SIZE = 10000
SYNC_TARGET_STEPS = 100 
MAX_STEPS = 50         # 增加步數限制給予更多探索空間

# ==========================================
# 網路組件 (Noisy + Dueling)
# ==========================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters(); self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                                        self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return nn.functional.linear(x, self.weight_mu, self.bias_mu)

class RainbowNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super().__init__()
        in_dim = np.prod(input_shape)
        self.action_size = action_size
        self.feature = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.value_stream = nn.Sequential(NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, N_ATOMS))
        self.advantage_stream = nn.Sequential(NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, action_size * N_ATOMS))
        self.register_buffer("support", torch.linspace(V_MIN, V_MAX, N_ATOMS))

    def forward(self, x):
        features = self.feature(x.view(x.size(0), -1))
        value = self.value_stream(features).view(-1, 1, N_ATOMS)
        advantage = self.advantage_stream(features).view(-1, self.action_size, N_ATOMS)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return nn.functional.softmax(q_atoms, dim=-1)

    def get_q_values(self, x):
        dist = self.forward(x)
        return torch.sum(dist * self.support, dim=2)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

# ==========================================
# Lightning 訓練模組
# ==========================================
class RainbowLitModule(pl.LightningModule):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.policy_net = RainbowNet(env.board.render_np().shape, 4)
        self.target_net = RainbowNet(env.board.render_np().shape, 4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.buffer = deque(maxlen=MEMORY_SIZE)
        self.n_step_buffer = deque(maxlen=N_STEPS)
        self.reward_history, self.total_steps = [], 0
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        if GAME_MODE == 'static': self.env.initGridStatic()
        elif GAME_MODE == 'player': self.env.initGridPlayer()
        else: self.env.initGridRand()
        
        # 修正：手動修改環境獎勵(暫時性)以維持收斂穩定，或在此處計算
        state = self.env.board.render_np()
        done, ep_reward, step_count = False, 0, 0
        
        # Epsilon Decay: 前 250 回合從 1.0 降到 0.1
        epsilon = max(0.1, 1.0 - (self.current_epoch / 250))
        
        while not done and step_count < MAX_STEPS:
            # 混合探索機制
            if random.random() < epsilon:
                action_idx = random.randint(0, 3)
            else:
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_idx = torch.argmax(self.policy_net.get_q_values(state_t)).item()
            
            self.env.makeMove(['u','d','l','r'][action_idx])
            raw_r = self.env.reward()
     
            
            next_s = self.env.board.render_np()
            done = True if raw_r == 20 else False
            
            # Multi-step 處理
            self.n_step_buffer.append((state, action_idx, raw_r, next_s, done))
            if len(self.n_step_buffer) == N_STEPS:
                r_n, s_n, d_n = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
                for i, transition in enumerate(self.n_step_buffer):
                    r_n += (GAMMA ** i) * transition[2]
                    if transition[4]: s_n, d_n = transition[3], True; break
                self.buffer.append((self.n_step_buffer[0][0], action_idx, r_n, s_n, d_n))

            if len(self.buffer) > BATCH_SIZE:
                opt = self.optimizers()
                minibatch = random.sample(self.buffer, BATCH_SIZE)
                sb = torch.tensor([m[0] for m in minibatch], dtype=torch.float).to(self.device)
                ab = torch.tensor([m[1] for m in minibatch], dtype=torch.long).to(self.device)
                rb = torch.tensor([m[2] for m in minibatch], dtype=torch.float).to(self.device)
                nb = torch.tensor([m[3] for m in minibatch], dtype=torch.float).to(self.device)
                db = torch.tensor([m[4] for m in minibatch], dtype=torch.float).to(self.device)

                with torch.no_grad():
                    next_a = torch.argmax(self.policy_net.get_q_values(nb), dim=1)
                    next_dist = self.target_net(nb)[range(BATCH_SIZE), next_a]
                    Tz = rb.view(-1, 1) + (GAMMA**N_STEPS) * (1 - db.view(-1, 1)) * self.policy_net.support.view(1, -1)
                    Tz = Tz.clamp(V_MIN, V_MAX)
                    b = (Tz - V_MIN) / ((V_MAX - V_MIN) / (N_ATOMS - 1))
                    l, u = b.floor().long(), b.ceil().long()
                    target_dist = torch.zeros(BATCH_SIZE, N_ATOMS).to(self.device)
                    batch_idx_range = torch.arange(BATCH_SIZE).to(self.device)
                    for j in range(N_ATOMS):
                        target_dist.index_put_((batch_idx_range, l[:, j]), next_dist[:, j] * (u[:, j].float() - b[:, j]), accumulate=True)
                        target_dist.index_put_((batch_idx_range, u[:, j]), next_dist[:, j] * (b[:, j] - l[:, j].float()), accumulate=True)

                curr_dist = self.policy_net(sb)[range(BATCH_SIZE), ab]
                loss = -(target_dist * curr_dist.log().clamp(-10, 10)).sum(dim=1).mean()
                opt.zero_grad(); self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0); opt.step()
                
                self.policy_net.reset_noise(); self.target_net.reset_noise()
                self.total_steps += 1
                if self.total_steps % SYNC_TARGET_STEPS == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            state, ep_reward, step_count = next_s, ep_reward + raw_r, step_count + 1
        
        if self.lr_schedulers(): self.lr_schedulers().step()
        self.reward_history.append(ep_reward)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        return [optimizer], [optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)]

# ==========================================
# 執行
# ==========================================
if __name__ == "__main__":
    model = RainbowLitModule(Gridworld(size=4, mode=GAME_MODE))
    trainer = pl.Trainer(max_epochs=EPISODES, logger=False, enable_checkpointing=False, devices=1)
    trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader([0]))

    # 繪圖：固定 Y 軸與規格
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