import os, sys, pygame, argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("checkpoints", exist_ok=True)

from game import FlappyBird
from models import Population, POP_SIZE
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, default=200)
parser.add_argument("--headless", action="store_true", help="skip rendering (faster)")
args = parser.parse_args()

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
clock = pygame.time.Clock() if not args.headless else None
best_ever = 0
stage = 0
fps = 60
gap, spd = 220, 2

log = Logger("logs/neuroevo.csv", [
    "gen", "score", "best_ever", "mean_fit", "gap", "speed", "stage"
])

for gen in range(1, args.generations + 1):
    for i, (req, g, s) in enumerate(DIFFICULTY):
        if best_ever >= req and i > stage:
            stage, gap, spd = i, g, s
            print(f"  >> harder now: gap={gap} speed={spd}")

    game = FlappyBird(n=POP_SIZE, headless=args.headless)
    game.gap = gap
    game.speed = spd
    game.reset()

    while True:
        if not args.headless:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pop.save("checkpoints/best_bird.pth", game.fit)
                    log.close()
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN:
                    # arrow keys to speed up / slow down the sim
                    if ev.key == pygame.K_UP: fps = min(fps + 30, 300)
                    if ev.key == pygame.K_DOWN: fps = max(fps - 30, 15)

        if not game.step(pop.get_actions(game)):
            break
        if not args.headless:
            game.render(gen=gen, best_score=best_ever, best_idx=0)
            clock.tick(fps)

    if game.score > best_ever:
        best_ever = game.score
        pop.save("checkpoints/best_bird.pth", game.fit)

    mean_fit = sum(game.fit) / len(game.fit)
    log.log({
        "gen": gen,
        "score": game.score,
        "best_ever": best_ever,
        "mean_fit": round(mean_fit, 2),
        "gap": gap,
        "speed": spd,
        "stage": stage,
    })

    w = pop.evolve(game.fit)
    print(f"gen {gen:>3} | score {game.score:>3} | best {best_ever:>3} | fit {game.fit[w]:.1f}")

log.close()
if not args.headless:
    pygame.quit()
