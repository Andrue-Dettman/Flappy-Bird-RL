import sys, pygame, torch

sys.path.insert(0, ".")
from game import FlappyBird
from models import Brain

path = "checkpoints/best_bird.pth"
try:
    brain = Brain()
    brain.load_state_dict(torch.load(path, weights_only=True))
except FileNotFoundError:
    print("no model yet — run train.py first")
    sys.exit(1)

game = FlappyBird()
game.reset()
clock = pygame.time.Clock()
best, n = 0, 0

while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: pygame.quit(); sys.exit()

    if not game.step([brain.decide(game.state(0))]):
        n += 1; best = max(best, game.score)
        print(f"game {n} | score {game.score} | best {best}")
        game.reset()
    game.render(best_score=best)
    clock.tick(60)
