# 強化學習HW3報告：DQN 演算法演進與 Gridworld 實踐

## 1. 作業目標
本作業旨在透過實作與改進深度 Q 網路（Deep Q-Network, DQN），解決一個經典的網格世界（Gridworld）導航問題。我們從最基礎的 Naive DQN 開始，逐步引入經驗回放（Experience Replay）、目標網路（Target Network）、Double DQN 與 Dueling DQN 等進階架構，最後結合 PyTorch Lightning 框架與多種訓練技巧（Training Tips），目標是讓 Agent 能在高度隨機的環境中穩定收斂並取得正向獎勵。

## 2. 背景設定與環境介紹
### 2.1 Gridworld 環境
本實驗使用的環境為 4x4 的網格空間(根據老師提供的連結網址的模板為架構修改reward值以更好觀察收斂狀況)，包含以下元件：
- **P (Player)**：Agent 的位置。
- **+ (Goal)**：終點，取得後獲得 **+20** 獎勵並結束回合。
- **- (Pit)**：陷阱，掉入後獲得 **-50** 的懲罰並繼續移動。
- **W (Wall)**：障礙物，無法通行。
- **每步懲罰**：每移動一步會獲得 **-1** 的獎勵，用以鼓勵 Agent 尋找最短路徑。

環境提供三種模式：
1. **Static Mode**：所有元件位置固定，是最基礎的測試場景。
2. **Player Mode**：僅玩家位置隨機產生，目標與陷阱位置固定。
3. **Random Mode**：所有元件（玩家、目標、陷阱、牆）的位置在每次初始化時皆隨機產生，難度最高。

### 2.2 固定模型與超參數設定
在基礎實驗中，我們設定了以下固定參數以利對照：
- **動作空間**：4 (上、下、左、右)
- **輸入維度**：4x4x4 (四個物件層的 One-hot 矩陣)
- **基礎網路結構**：64 -> 32 -> 4 (Linear + ReLU)
- **優化器**：Adam (LR=0.001)
- **折扣因子 (Gamma)**：0.9
- **探索率 (Epsilon)**：固定 0.1

---

## 3. 實驗過程與結果分析

### 3.1 Basic DQN(HW3-1)

#### Naive DQN 原理

深度 Q 網路（Deep Q-Network, DQN）是強化學習領域的一個里程碑，其核心思想是利用**深度神經網路（Deep Neural Network**來取代傳統 Q-Learning 中的 **Q 表格（Q-Table）**。在傳統方法中，我們需要記錄每一種狀態與動作組合的價值，但當面對如 Gridworld 這樣狀態空間較大或具備隨機性的環境時，表格法會面臨「維度災難」而失效。

DQN 的運作邏輯可以拆解為以下三個關鍵維度：

1. **神經網路作為函數近似器 (Function Approximator)**：
   在 DQN 中，神經網路扮演著一個「預測器」的角色。給予環境的狀態 $s$（在本作業中為 4x4x4 的矩陣），網路會輸出該狀態下所有可能動作（上、下、左、右）的預估價值，即 $Q(s, a; \theta)$，其中 $\theta$ 為網路權重。這使得模型具備了**泛化能力**，即使是沒見過的狀態，網路也能根據過往經驗推測出合理的動作價值。

2. **時序差分學習 (Temporal Difference Learning)**：
   DQN 遵循貝爾曼方程式（Bellman Equation）的精神進行更新。其學習目標（TD Target）定義為當前獲得的即時獎勵 $r$ 加上對未來狀態價值的預期：
   $$Y^{DQN} = r + \gamma \max_{a'} Q(s', a'; \theta)$$
   其中 $\gamma$ 為折扣因子，代表我們對未來獎勵的重視程度。



3. **損失函數與更新機制**：
   在 Naive DQN 的設定中，模型透過最小化「預測 Q 值」與「目標 Q 值（TD Target）」之間的**均方誤差（MSE）**來訓練：
   $$Loss = ( (r + \gamma \max_{a'} Q(s', a'; \theta)) - Q(s, a; \theta) )^2$$
   
   **Naive 版本的主要缺陷：**
   在 3-1 的實作中，Naive 版本採取「單步更新」策略，即 Agent 每執行一個動作 $(s, a, r, s')$，就立刻對網路進行一次反向傳播。這種方式存在嚴重的穩定性問題：
   - **樣本高度相關**：相鄰的動作與狀態極其相似，導致梯度下降的方向過於單一，網路容易發生災難性遺忘。
   - **目標不穩定**：由於沒有使用獨立的目標網路，更新權重 $\theta$ 的同時，計算 TD Target 的參考標準也在變動。這就像一個射箭手在瞄準時，靶心（Target）卻隨著弓箭手的動作而隨機移動，導致收斂極其困難。

#### Naive DQN 實驗結果
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/bdbe0503-e48f-4e7c-8dd2-d3e43df3be92" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c88194e6-1da8-4059-b480-d51d97b25bac" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c0b5d0a9-8c94-4e5f-acd0-cd0590d32e9c" />


**結果說明：**
- **Static 模式**：幾乎無法收斂，獎勵值多保持在 -30 左右。這是因為 Agent 陷入局部最佳解或在牆壁間徘徊，最終觸發了 30 步的強制停止限制（30 步 $\times$ -1/step = -30）。
- **Player 模式**：相較於 Static，有時會出現些微的收斂趨勢。原因在於玩家位置隨機化雖然增加了難度，但也變相增加了 Agent 觸碰不同狀態的機率（即被動探索），讓網路有機會學到更多元的路徑特徵。
- **Random 模式**：完全無法收斂，分數持續低迷且震盪劇烈。

#### 經驗回放 (Experience Replay) 原理
為了優化 Naive DQN，我們引入了「經驗回放池」。Agent 將經驗 $(s, a, r, s')$ 存入一個 Buffer 中，訓練時隨機抽取一個 Batch 的資料進行更新。這能有效打破資料間的相關性，並提高資料利用率。

#### 經驗回放實驗結果分析
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/6e983043-1baf-408c-b8bf-44ef9bd3ffaf" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/bba2bc08-7a0a-4fcb-b33b-495db360f164" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/33bff433-3c1c-4183-b0bc-15ffbc7f6c38" />


**結果說明：**
引入經驗回放後，**Static** 與 **Player** 模式已經能夠達到收斂，Agent 開始學會走向終點。然而，訓練過程仍有明顯震盪。而在 **Random** 模式下，由於環境狀態組合過於龐大，單純的經驗回放仍不足以穩定學習。

---

### 3.2 Enhanced DQN(HW3-2)

#### Double DQN (DDQN)
**原理**：傳統 DQN 容易出現「過度估計（Overestimation）」Q 值的問題。DDQN 透過拆分「動作選擇」與「數值評估」來解決：利用 Policy Net 決定動作，再利用 Target Net 計算該動作的 Q 值，進而提供更精確的更新目標。

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/eb781526-5640-40d0-ad4d-3933f136ddc8" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/1ff77103-b3d3-47b0-9663-4957d4093573" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/b1925ebd-7ab0-4c55-b714-5768bb0271aa" />


#### Dueling DQN
**原理**：將神經網路輸出拆解為兩個支流：**狀態價值 (Value)** 與 **動作優勢 (Advantage)**。這樣做的好處是即便 Agent 沒試過某個動作，也能透過對該狀態的整體價值評估來學習。

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/9b641d24-6cac-4a73-958f-03ae973766e9" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/920e895f-ed1b-4c69-9dd1-ee200c6e4d87" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/d2839ec1-01c4-4bbd-820f-abb0fcef59c6" />


#### 兩者結合：Dueling Double DQN
將上述兩者結合後，Agent 同時具備了抑制過度估計與精細化狀態評估的能力。

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/51ad5c59-d5b3-4391-a172-d89469440a9e" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/473a9699-b03c-4915-8793-4b13ddd6acb3" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/49aabc9f-ba20-4b63-a7c4-87282d75d124" />


**結果說明：**
比起普通 DQN，結合後的版本在static與player模式下展現了以下優點：
1. **更早收斂**：在較少的回合數內就達到了高分區域。
2. **震盪更少**：曲線相對平滑，代表策略更新較穩定。
3. **安全性提升**：Agent 更懂得避開陷阱，獎勵值很少掉到 -50 以下的極端低點，代表其對風險的判斷更為精準。  
`然而面對Random模式仍然沒辦法有效的學習`
---

### 3.3 Enhance DQN with Training Tips(HW3-3)

#### 導入 PyTorch Lightning
改用 PyTorch Lightning 框架後，我們獲得了以下好處：
- **程式碼結構化**：將環境交互、網路更新與硬體加速（GPU）邏輯分離，減少 Boilerplate 程式碼。
- **內建優化**：更容易實現梯度裁剪（Gradient Clipping）與學習率調度器（LR Scheduler）。

#### 訓練技巧 (Training Techniques)
在此階段我們加入了：
1. **Epsilon Decay**：從 1.0 逐漸降至 0.1，讓 Agent 在早期充分探索。
2. **學習率排程**：透過 Cosine Annealing 讓學習率隨訓練進度呈餘弦曲線下降，使模型在前期能大步探索，而在後期則精細微調以趨於穩定。
3. **梯度裁剪**：設定梯度強制的上限閾值（Threshold），防止因單次獎勵回傳（如踩到陷阱）產生過大的更新力道而摧毀已學會的網路權重。

#### 初步實驗結果 
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/d114364e-3222-41e6-baed-e16481b06b03" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/fb4f8569-ab45-4869-81f4-71504f4439e2" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c89e39c8-d370-4732-83c8-134a39428837" />


**結果說明：**
對於簡單的 Static 與 Player 模式，雖然與 3-2 相比提升不明顯，這些技巧讓收斂但穩定性更高。然而，**Random 模式** 雖然開始能維持在較高的分數（約 -100 到 -200 之間），不再像先前頻繁掉出頁面底端，但仍未真正收斂到正分。

#### 參數調整與網路擴張 (pytorch_lighting_new)
針對 Random 模式不收斂的問題，我們進行了以下診斷與改進：
1. **回合數不足**：Random 模式狀態空間巨大，500 回合不足以遍歷足夠多的組合。
2. **網路表達能力弱**：原先的 64->32 架構無法紀錄複雜的地圖特徵。
3. **探索時間不夠**：MAX_STEPS 太少可能導致 Agent 還沒走到終點就被強制結束。

**改動細節：**
- **回合數**：增加至 **3000 EPISODES**。
- **神經網路**：擴大為 **256 -> 128 -> 64**。
- **探索空間**：MAX_STEPS 增加至 **50**，Batch Size 提高至 **64**。

#### 最終優化結果分析
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/46814731-9abd-42fa-a674-ab4c0c651f71" />


**結果說明：**
從長遠來看，Random 模式終於展現出明顯的收斂趨勢。雖然仍有波動，但隨著學習率下降與經驗累積，**平均獎勵線在後期已經突破 0 分大關**，代表 Agent 在隨機地圖中找到終點的機率已遠高於踩到陷阱或超時，證明了擴大容量與延長訓練時間對複雜環境的必要性。

### 3.4 Rainbow DQN(HW3-4)

#### Rainbow 技術介紹
在 `3-4Ranbow_DQN.py` 中，我們整合了：
1. **Noisy Networks**：參數化雜訊取代 $\epsilon$-greedy，提供更穩定的探索。
2. **Distributional RL (C51)**：學習獎勵的機率分佈（Atoms）而非單一期望值。
3. **Multi-step ($n$-step)**：預測未來 3 步的獎勵，加速信號回傳。

#### 實驗結果分析：為什麼 Rainbow 反而失敗？
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/77da397b-0b99-4db7-92fe-7051acaff22d" />


**觀察：**
儘管使用了最先進的技術與同樣的 256-128-64 架構，Rainbow DQN 在 Random 模式下的表現卻不如預期的 3-3 版本穩定，橘線平均值遲遲無法穩定位於 0 分以上，且藍線 Raw Reward 震盪極其劇烈。

**失敗原因分析與探討：**
1. **過度擬合與複雜度失衡**：在 4x4 的小規模 Gridworld 中，C51 (Categorical DQN) 試圖將獎勵映射到 51 個分佈點上。對於這種「獎勵極端（非 -1 即 20/50）」且空間狹小的環境，分佈式學習反而引入了過多的計算雜訊與不穩定性，不如 3-3 中的標量 Q 值（Scalar Q-value）直接預測來得有效。
2. **$N$-step 的副作用**：在高度隨機的 Random 模式下，環境每回合都不同，3-step 的累積獎勵會包含更多不確定性，導致更新目標的方差（Variance）過大，難以穩定權重。
3. **超參數敏感度**：Rainbow 包含過多超參數（如 $V_{min}/V_{max}$、Noisy 初始標準差等）。在嘗試過多組參數設定後，模型依然沒能收斂，這反映了 Rainbow 雖然強大，但需要極高成本的參數微調才能適應特定環境。
4. **效能與時間限制**：Rainbow 的每一輪更新計算量遠大於一般 DQN。礙於個人電腦硬體效能與實驗時間成本，無法進行萬次以上的長時訓練。在相同的 3000 回合內，結構較精簡的 Double Duel DQN 反而更容易在隨機環境中找到穩定的收斂路徑。

## 4. 完整總結與實驗洞察

本作業透過一系列的實驗，完整見證了深度 Q 網路從最基礎的 Naive 架構演進至現代強化學習技術組合的過程。在針對 Gridworld 環境的測試中，我們得出以下關鍵結論與深度觀察：

### 4.1 演算法演進的有效性
1.  **基礎改進的必要性**：從 Naive DQN 到引入 **經驗回放（Experience Replay）** 與 **目標網路（Target Network）**，是模型從「完全無法學習」轉變為「具備基礎收斂能力」的分水嶺。這證明了打破樣本相關性與穩定訓練目標在強化學習中的核心地位。
2.  **架構優化的紅利**：**Double DQN** 與 **Dueling DQN** 的結合在所有測試模式中均表現優異。其不僅有效抑制了 Q 值的過度估計，更透過 Value 與 Advantage 的分流，使 Agent 在複雜狀態下仍能保持對「風險」的高度敏感，大幅降低了誤入陷阱（-50 分）的機率。

### 4.2 隨機環境下的「規模效應」與「簡約法則」
在難度最高的 **Random 模式** 下，實驗結果揭示了一個重要的現象：**適度的模型容量與充足的訓練時間，其重要性往往高於算法的複雜度。**

* **3-3 版本的成功關鍵**：在導入 PyTorch Lightning 並將網路層擴張至 **256 -> 128 -> 64** 且訓練延長至 **3000 回合** 後，Dueling Double DQN 展現了穩健的收斂趨勢，平均獎勵最終突破 0 分。這說明對於高隨機性的狀態空間，模型需要足夠的「記憶體容量」來紀錄地圖特徵，並需要足夠的探索步數來覆蓋稀疏的獎勵信號。
* **3-4 Rainbow DQN 的挫敗與反思**：儘管 Rainbow 整合了 C51、Noisy Nets 與 N-step 等頂尖技術，但在本實驗的 4x4 小規模網格中，表現反而不如 3-3 版本穩定。
    * **分佈式學習的雜訊**：C51 試圖將簡單的獎勵（-1, 20, -50）映射到 51 個原子分佈上，在狹小空間內反而產生了過多的計算開銷與數值波動，導致收斂變慢。
    * **$n$-step 的方差問題**：在每局地圖皆隨機的 Random 模式下，3-step 的累積獎勵包含了過高的環境不確定性，導致梯度的方差過大，權重難以定型。

### 4.3 實驗限制與自我檢討
在本次作業的後期，我們面臨了顯著的技術瓶頸，強化學習的訓練本質上是高度消耗**時間成本**與**超參數微調工程**的。

* **效能限制**：Rainbow DQN 的運算複雜度極高，在同樣的 3000 回合內，其權重更新的收斂速度未能追上環境的隨機變動率。若要發揮 Rainbow 的真正潛力，可能需要上萬回合的訓練以及更精細的超參數網格搜索（Grid Search）。
* **結論**：在受限的時間與計算資源下，**「擴大規模後的經典 Dueling Double DQN」** 是解決此類網格導航問題的最優解。它在模型複雜度與收斂穩定性之間達到了極佳的平衡，成功讓 Agent 在充滿變數的隨機世界中，找到一條通往勝利（+20）的穩定路徑。

