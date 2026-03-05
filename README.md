# Flappy Bird AI: Neuroevolution & Deep Q-Network

Training an AI to play Flappy Bird using two different approaches: evolutionary neural networks and deep reinforcement learning.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-latest-orange) ![Pygame](https://img.shields.io/badge/Pygame-latest-green)

---

## Demo

*(add a gif here after recording with any screen recorder)*

![Demo](output/demo.gif)

Training curves (generated after running):

![Neuroevo](output/figures/neuroevo.png)
![DQN](output/figures/dqn.png)

---

## How it works

### Approach 1: Neuroevolution (genetic algorithm)

Runs 50 birds in parallel each generation. Each bird has a small neural net (5 inputs → 32 → 16 → 1 output) that decides whether to flap. After every generation, the top 5 networks survive unchanged and the rest are mutated copies. No gradient descent — just selection pressure and random noise.

The network is intentionally small: with 50 agents and ~700 parameters each, the evolutionary search is tractable. A bigger network would make the weight space too large to explore meaningfully through mutation alone.

Also has a curriculum — starts with wide pipe gaps (220px) and progressively narrows them as the agent improves. Without this the birds struggle to learn anything early on, because they die before getting useful fitness signal.

```
Observation [5D] → BirdBrain (5→32→16→1) → flap probability → threshold at 0.5
                              ↑
                 weights evolved via selection + mutation
```

### Approach 2: DQN (work in progress)

Standard Deep Q-Network with experience replay and a target network. Single bird, one episode at a time. The target network gets synced every 500 steps to keep Q-value estimates stable.

Reward shaping:
- **+0.01** per frame alive (small survival bonus — breaks tie when all birds die at score 0)
- **+1.0** per pipe passed
- **-1.0** on death

Without the survival bonus, early training is almost entirely -1.0 rewards with no signal about what went wrong. The per-frame bonus at least rewards birds that stay airborne longer.

```
Observation [5D] → BirdBrain (5→32→16→2) → Q-values → argmax → action
                              ↑
                 trained via TD error + experience replay
```

---

## Quick start

```bash
git clone https://github.com/Andrue-Dettman/Flappy-Bird-RL.git
cd Flappy-Bird-RL
pip install -r requirements.txt

# train via neuroevolution (renders by default, arrow keys to speed up/slow down)
python train.py

# train via DQN (headless — faster)
python train_dqn.py

# watch the best neuroevo model play
python watch.py

# play yourself
python play.py

# generate training plots (after training)
python plot_training.py

# run tests
python -m pytest tests/
```

---

## Project structure

```
flappy-bird-rl/
├── game/
│   └── flappy_bird.py       # game environment (physics, rendering, state)
├── models/
│   ├── genetic_agent.py     # neuroevolution: Population, Brain, mutation
│   └── dqn_agent.py         # DQN: BirdBrain, ReplayMemory, DQNAgent
├── train.py                 # neuroevolution training loop
├── train_dqn.py             # DQN training loop
├── watch.py                 # watch a trained model play
├── play.py                  # manual play
├── plot_training.py         # generate training plots from logs
├── config.py                # DQN hyperparameters
├── logger.py                # CSV training logger
├── logs/                    # training logs (CSV)
├── output/figures/          # training plots
└── tests/                   # unit tests
```

---

## Design decisions

**Why neuroevolution instead of NEAT or similar?**
Wanted full control over the evolutionary mechanics — tournament selection, elitism, Gaussian mutation. NEAT adds a lot of complexity (speciation, crossover) that isn't necessary for a problem this constrained. Implementing it from scratch also meant I actually understood every part.

**Why such a small network?**
Neuroevolution searches the weight space directly via random perturbation. With 50 agents and ~50 parameters per network (5→32→16→1), the search is tractable. I started with 64 neurons in the first layer and it worked fine but trained slower with no real benefit.

**Why curriculum learning?**
Without it, the first ~30 generations of birds die almost immediately — the gaps start too small and they never accumulate meaningful fitness signal. Starting with wide gaps (220px, basically impossible to fail) lets the population learn the basic flap-or-don't timing before being challenged.

**Why implement both approaches?**
Neuroevolution and gradient-based RL make fundamentally different tradeoffs. Neuroevo is sample-inefficient (runs all 50 birds every generation) but doesn't need reward shaping — the fitness function is just pipes passed. DQN is much more sample-efficient but sensitive to reward design and hyperparameters. Interesting to compare.

---

Andrue Dettman — UW-Madison CS / Data Science / Economics
