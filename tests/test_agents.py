import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pygame
pygame.init()

from models.genetic_agent import Brain, Population, POP_SIZE
from models.dqn_agent import DQNAgent, BirdBrain


def test_brain_output_shape():
    brain = Brain()
    state = np.array([0.5, 0.1, 0.3, -0.1, 0.8], dtype=np.float32)
    action = brain.decide(state)
    assert action in (0, 1), f"expected 0 or 1, got {action}"


def test_brain_forward():
    brain = Brain()
    x = torch.FloatTensor([[0.5, 0.1, 0.3, -0.1, 0.8]])
    out = brain(x)
    assert out.shape == (1, 1)
    assert 0.0 <= out.item() <= 1.0  # sigmoid output


def test_population_size():
    pop = Population()
    assert len(pop.brains) == POP_SIZE


def test_population_evolve():
    pop = Population()
    # fake fitness — first brain is the best
    fitnesses = [float(i) for i in range(POP_SIZE)]
    winner = pop.evolve(fitnesses)
    assert winner == POP_SIZE - 1  # highest fitness index
    assert len(pop.brains) == POP_SIZE


def test_dqn_action_shape():
    agent = DQNAgent()
    state = np.array([0.5, 0.1, 0.3, -0.1, 0.8], dtype=np.float32)
    action = agent.pick_action(state)
    assert action in (0, 1)


def test_dqn_memory_accumulates():
    agent = DQNAgent()
    state = np.zeros(5, dtype=np.float32)
    for _ in range(10):
        agent.step(state, 0, 0.01, state, False)
    assert len(agent.memory) == 10


def test_dqn_birdbrain_output():
    net = BirdBrain()
    x = torch.FloatTensor([[0.5, 0.1, 0.3, -0.1, 0.8]])
    out = net(x)
    assert out.shape == (1, 2)  # Q values for [no_flap, flap]


if __name__ == "__main__":
    test_brain_output_shape()
    test_brain_forward()
    test_population_size()
    test_population_evolve()
    test_dqn_action_shape()
    test_dqn_memory_accumulates()
    test_dqn_birdbrain_output()
    print("all agent tests passed")
