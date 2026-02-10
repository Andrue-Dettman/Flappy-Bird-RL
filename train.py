import os, sys, pygame

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("checkpoints", exist_ok=True)

from game import FlappyBird
from models import Population, POP_SIZE

# (min_score_to_unlock, pipe_gap, pipe_speed)
# starts wide so the birds can actually survive long enough to learn,
# then tightens up once they prove they can handle it
DIFFICULTY = [
    (0,  220, 2),
    (3,  200, 2.5),
    (5,  180, 2.5),
    (10, 160, 3),
    (20, 150, 3),
]

pop = Population()
clock = pygame.time.Clock()
best_ever = 0
stage = 0
fps = 60
gap, spd = 220, 2

for gen in range(1, 201):
    for i, (req, g, s) in enumerate(DIFFICULTY):
        if best_ever >= req and i > stage:
            stage, gap, spd = i, g, s
            print(f"  >> harder now: gap={gap} speed={spd}")

    game = FlappyBird(n=POP_SIZE)
    game.gap = gap
    game.speed = spd
    game.reset()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pop.save("checkpoints/best_bird.pth", game.fit)
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                # arrow keys to speed up / slow down the sim
                if ev.key == pygame.K_UP: fps = min(fps + 30, 300)
                if ev.key == pygame.K_DOWN: fps = max(fps - 30, 15)

        if not game.step(pop.get_actions(game)):
            break
        game.render(gen=gen, best_score=best_ever, best_idx=0)
        clock.tick(fps)

    if game.score > best_ever:
        best_ever = game.score
        pop.save("checkpoints/best_bird.pth", game.fit)
        # print(f"  new best! saved")

    w = pop.evolve(game.fit)
    print(f"gen {gen:>3} | score {game.score:>3} | best {best_ever:>3} | fit {game.fit[w]:.1f}")

pygame.quit()
