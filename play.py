import sys, pygame

sys.path.insert(0, ".")
from game import FlappyBird

game = FlappyBird()
game.reset()
clock = pygame.time.Clock()
best = 0

while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: pygame.quit(); sys.exit()

    flap = int(pygame.key.get_pressed()[pygame.K_SPACE])
    if not game.step([flap]):
        best = max(best, game.score)
        print(f"score {game.score} | best {best}")
        game.reset()
    game.render(best_score=best)
    clock.tick(60)
