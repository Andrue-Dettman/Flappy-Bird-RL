import pygame, random
import numpy as np

W, H = 500, 600
GROUND_Y = 550


class FlappyBird:
    def __init__(self, n=1, headless=False):
        self.n = n
        self.headless = headless
        self.gap = 150
        self.speed = 3

        if not headless:
            pygame.init()
            self.scr = pygame.display.set_mode((W, H))
            pygame.display.set_caption("flappy bird neuroevolution")
            self.font = pygame.font.Font(None, 52)
            self.hud_font = pygame.font.Font(None, 22)

        self.reset()

    def reset(self):
        mid = GROUND_Y // 2
        self.ys = [float(mid)] * self.n
        self.vs = [0.0] * self.n
        self.alive = [True] * self.n
        self.fit = [0.0] * self.n
        self.pipes = [self._rand_pipe()]
        self.score = 0
        self.ticks = 0

    def _rand_pipe(self):
        return [W, random.randint(80, GROUND_Y - 80 - self.gap)]

    def step(self, actions):
        self.ticks += 1
        for i in range(self.n):
            if not self.alive[i]: continue
            if actions[i]: self.vs[i] = -8
            self.vs[i] += 0.5
            self.ys[i] += self.vs[i]
            self.fit[i] += 0.1

        for p in self.pipes: p[0] -= self.speed

        if self.pipes[0][0] < -60:
            self.pipes.pop(0)
            self.pipes.append(self._rand_pipe())
            self.score += 1
            for i in range(self.n):
                if self.alive[i]: self.fit[i] += 10

        for i in range(self.n):
            if not self.alive[i]: continue
            y = self.ys[i]
            if y < 0 or y > GROUND_Y:
                self.alive[i] = False
                continue
            for px, pg in self.pipes:
                # 60 is bird x pos, 12 is radius
                if 60 + 12 > px and 60 - 12 < px + 60:
                    if y - 12 < pg or y + 12 > pg + self.gap:
                        self.alive[i] = False
                        break

        return any(self.alive)

    def state(self, i):
        px, pg = self.pipes[0]
        y = self.ys[i]
        v = self.vs[i]
        # TODO: maybe normalize v differently? /10 feels arbitrary
        return np.array([y/GROUND_Y, v/10, (px-60)/W, (pg - y)/GROUND_Y,
                         (pg + self.gap - y)/GROUND_Y])

    def render(self, gen=0, best_score=0, best_idx=None):
        if self.headless: return
        scr = self.scr
        scr.fill((135, 206, 235))

        # pipes
        for px, pg in self.pipes:
            bot = pg + self.gap
            pygame.draw.rect(scr, (80,190,60), (px, 0, 60, pg-18))
            pygame.draw.rect(scr, (55,155,40), (px-5, pg-18, 70, 18))
            pygame.draw.rect(scr, (80,190,60), (px, bot+18, 60, GROUND_Y-bot-18))
            pygame.draw.rect(scr, (55,155,40), (px-5, bot, 70, 18))

        # draw all the birds
        for i in range(self.n):
            by = int(self.ys[i])
            if not self.alive[i]:
                # faded ghost for dead birds
                s = pygame.Surface((20,20), pygame.SRCALPHA)
                pygame.draw.circle(s, (180,180,180,35), (10,10), 10)
                scr.blit(s, (50, by-10))
            else:
                # red = elite from last gen, yellow = everyone else
                col = (230,55,55) if i == best_idx else (255,200,40)
                pygame.draw.circle(scr, col, (60, by), 12)
                # lil eye
                pygame.draw.circle(scr, (255,255,255), (65, by-4), 4)
                pygame.draw.circle(scr, (20,20,20), (66, by-4), 2)

        pygame.draw.rect(scr, (195,170,90), (0, GROUND_Y, W, H - GROUND_Y))
        pygame.draw.line(scr, (160,140,70), (0, GROUND_Y), (W, GROUND_Y), 2)

        # score centered at top
        t = self.font.render(str(self.score), True, (255,255,255))
        scr.blit(t, (W//2 - t.get_width()//2, 18))

        # bottom left info
        alive_n = sum(self.alive)
        info = [f"gen {gen}", f"alive {alive_n}/{self.n}", f"best {best_score}"]
        for j in range(len(info)):
            s = self.hud_font.render(info[j], True, (255,255,255))
            scr.blit(s, (8, GROUND_Y - 56 + j * 18))

        pygame.display.flip()
