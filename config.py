# hyperparameters — pulled into one place so I'm not hunting for magic numbers

# screen
W = 500
H = 600
GROUND_Y = 550

# neuroevolution
POP_SIZE = 50
ELITE_COUNT = 5
MUTATION_RATE = 0.2
MUTATION_STR = 0.5     # tried lower values but population converged too slowly

# DQN
MEMORY_SIZE = 50_000
LEARNING_RATE = 1e-3
EXPLORATION_START = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
TARGET_SYNC_EVERY = 500      # steps between target net syncs
RANDOM_FLAP_RATE = 0.07      # during random exploration, bias toward not flapping
WARMUP_STEPS = 1000          # fill buffer before training starts
TRAIN_EVERY_N_FRAMES = 4
GRAD_CLIP = 1.0
