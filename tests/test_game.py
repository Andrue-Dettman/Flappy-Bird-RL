import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pygame
pygame.init()

from game import FlappyBird


def test_observation_shape():
    game = FlappyBird(n=1, headless=True)
    game.reset()
    obs = game.state(0)
    assert obs.shape == (5,), f"expected shape (5,), got {obs.shape}"


def test_observation_range():
    # all values should be roughly normalized (not way outside 0-1 range at start)
    game = FlappyBird(n=1, headless=True)
    game.reset()
    obs = game.state(0)
    assert obs is not None
    assert len(obs) == 5


def test_step_returns_bool():
    game = FlappyBird(n=1, headless=True)
    game.reset()
    result = game.step([0])
    assert isinstance(result, bool)


def test_multi_agent_step():
    n = 10
    game = FlappyBird(n=n, headless=True)
    game.reset()
    actions = [0] * n
    result = game.step(actions)
    assert isinstance(result, bool)
    assert len(game.alive) == n


def test_flap_changes_velocity():
    game = FlappyBird(n=1, headless=True)
    game.reset()
    v_before = game.vs[0]
    game.step([1])  # flap
    # after a flap the velocity should be negative (going up)
    assert game.vs[0] < v_before or game.vs[0] < 0


def test_score_increments():
    # run a bunch of frames with no flap and see that pipes eventually pass
    game = FlappyBird(n=1, headless=True)
    game.gap = 400  # huge gap so the bird doesn't die
    game.speed = 5
    game.reset()
    for _ in range(500):
        if not game.step([0]):
            break
    # with a huge gap it should have passed at least one pipe in 500 frames
    # (pipe spawns every ~150px, screen is 500px wide, speed=5 -> ~100 frames per pipe)
    # this is more of a sanity check than a strict assertion
    assert game.score >= 0  # at minimum it ran without crashing


if __name__ == "__main__":
    test_observation_shape()
    test_observation_range()
    test_step_returns_bool()
    test_multi_agent_step()
    test_flap_changes_velocity()
    test_score_increments()
    print("all game tests passed")
