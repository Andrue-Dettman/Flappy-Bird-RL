import sys, pygame, torch, argparse

sys.path.insert(0, ".")
from game import FlappyBird

parser = argparse.ArgumentParser()
parser.add_argument("--dqn", action="store_true", help="load DQN model instead of neuroevo")
args = parser.parse_args()

if args.dqn:
    from models.dqn_agent import DQNAgent
    path = "checkpoints/best_dqn.pth"
    agent = DQNAgent()
    try:
        agent.load(path)
    except FileNotFoundError:
        print("no DQN model found — run train_dqn.py first")
        sys.exit(1)
    def get_action(game):
        return agent.pick_action(game.state(0))
else:
    from models import Brain
    path = "checkpoints/best_bird.pth"
    brain = Brain()
    try:
        brain.load_state_dict(torch.load(path, weights_only=True))
    except FileNotFoundError:
        print("no model yet — run train.py first")
        sys.exit(1)
    def get_action(game):
        return brain.decide(game.state(0))

game = FlappyBird()
game.reset()
clock = pygame.time.Clock()
best, n = 0, 0

while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: pygame.quit(); sys.exit()

    if not game.step([get_action(game)]):
        n += 1; best = max(best, game.score)
        print(f"game {n} | score {game.score} | best {best}")
        game.reset()
    game.render(best_score=best)
    clock.tick(60)
