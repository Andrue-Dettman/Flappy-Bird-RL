import os, sys, pygame

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("checkpoints", exist_ok=True)

from game import FlappyBird
from models.dqn_agent import DQNAgent
from logger import Logger

EPISODES = 1000

agent = DQNAgent()
log = Logger("logs/dqn.csv", ["episode", "score", "reward", "epsilon", "loss"])
clock = pygame.time.Clock()
best_score = 0

for ep in range(1, EPISODES + 1):
    game = FlappyBird(n=1, headless=True)
    state = game.state(0)
    total_reward = 0.0
    total_loss = 0.0
    loss_count = 0
    prev_score = 0

    while True:
        action = agent.pick_action(state)
        alive = game.step([action])

        pipe_passed = game.score > prev_score
        prev_score = game.score
        done = not alive

        # reward shaping: small alive bonus, big reward for pipe, penalty for death
        if done:
            reward = -1.0
        elif pipe_passed:
            reward = 1.0
        else:
            reward = 0.01

        next_state = game.state(0) if not done else state
        loss = agent.step(state, action, reward, next_state, done)

        if loss is not None:
            total_loss += loss
            loss_count += 1

        if agent.total_steps % agent.sync_every == 0:
            agent.sync_target()

        total_reward += reward
        state = next_state

        if done:
            break

    agent.decay_exploration()

    if game.score > best_score:
        best_score = game.score
        agent.save("checkpoints/best_dqn.pth")

    avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
    log.log({
        "episode": ep,
        "score": game.score,
        "reward": round(total_reward, 2),
        "epsilon": round(agent.exploration_rate, 4),
        "loss": round(avg_loss, 4),
    })

    if ep % 50 == 0:
        print(f"ep {ep:>4} | score {game.score:>3} | best {best_score:>3} | "
              f"eps {agent.exploration_rate:.3f} | loss {avg_loss:.4f}")

log.close()
print(f"\ndone. best score: {best_score}")
