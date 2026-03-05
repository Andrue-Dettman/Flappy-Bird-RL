# flappy bird RL

Trained an agent to play Flappy Bird using a genetic algorithm. Also started implementing DQN as a comparison — that's still in progress.

## how it works

The genetic approach runs 50 birds at once per generation. Each bird has a small neural net that decides whether to flap based on 5 inputs (position, velocity, distance to next pipe, etc.). After each generation the top 5 survive and the rest are mutated copies of them. It also has a curriculum — starts with wide pipe gaps and gradually makes it harder as the birds improve.

The DQN version (`models/dqn_agent.py`) uses experience replay and a target network. Still tuning the reward function on that one.

## running it

```
pip install -r requirements.txt

python train.py   # run the genetic trainer, arrow keys to speed up/slow down
python watch.py   # watch the best saved model play
python play.py    # play yourself
```

Requires a trained checkpoint to use `watch.py` — run `train.py` first.
