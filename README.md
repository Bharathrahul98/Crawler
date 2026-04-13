# 🧠 Crawler Game (Pygame + RL)

## 📌 Features
- 4 arms + 4 forearms
- 20 continuous actions
- 10 agents training
- Geometric reward
- Goal-based movement

---

## ⚙️ Setup in VS Code

### 1. Open folder in VS Code

### 2. Create virtual environment
python -m venv venv

Activate:
venv\Scripts\activate

---

### 3. Install dependencies
pip install -r requirements.txt

---

## ▶️ Run

### Train
python train.py

---

### Play
python play.py

---

## 🎮 Output
- Blue creature = agent
- Black lines = arms
- Light blue = forearms
- Green = goal

---

## 🧠 Explanation

Agent learns to:
- Align velocity toward goal
- Maintain direction
- Move using limb coordination

Reward is geometric:
reward = velocity_match × direction_alignment