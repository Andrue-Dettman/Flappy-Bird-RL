import torch, torch.nn as nn, numpy as np, copy
import config

POP_SIZE = config.POP_SIZE
ELITES = config.ELITE_COUNT
MUTATE_RATE = config.MUTATION_RATE
MUTATE_STR = config.MUTATION_STR


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        # started with 64 neurons but 32 trains faster and works just as well
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    def decide(self, state):
        with torch.no_grad():
            return int(self(torch.FloatTensor(state).unsqueeze(0)).item() > 0.5)


def _mutate(brain):
    """add random noise to some weights"""
    for p in brain.parameters():
        mask = torch.rand_like(p) < MUTATE_RATE
        p.data += mask * torch.randn_like(p) * MUTATE_STR


class Population:
    def __init__(self):
        self.brains = [Brain() for _ in range(POP_SIZE)]

    def get_actions(self, env):
        out = []
        for i in range(len(self.brains)):
            if env.alive[i]:
                out.append(self.brains[i].decide(env.state(i)))
            else:
                out.append(0)
        return out

    def evolve(self, fit):
        ranked = sorted(range(len(self.brains)), key=lambda i: fit[i], reverse=True)
        top = [copy.deepcopy(self.brains[i]) for i in ranked[:ELITES]]

        new = list(top)  # elites pass through untouched
        while len(new) < POP_SIZE:
            # pick a random elite, clone it, mutate the clone
            parent = copy.deepcopy(top[np.random.randint(ELITES)])
            _mutate(parent)
            new.append(parent)

        self.brains = new
        return ranked[0]  # return index of the winner

    def save(self, path, fit):
        best = max(range(len(self.brains)), key=lambda i: fit[i])
        torch.save(self.brains[best].state_dict(), path)

    def load(self, path):
        w = torch.load(path, weights_only=True)
        for b in self.brains:
            b.load_state_dict(w)
        for b in self.brains[1:]:
            _mutate(b)
