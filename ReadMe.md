# Learning about reinforcement learning using Hex

My goal is a Hex game that helps the opponent learn and play the game. The
opponent might have a disability or simply want to learn to play with hints.

Install the requirements then run

```
python train.py --help
usage: train.py [-h] [--size SIZE] [--seed SEED] [--train_games TRAIN_GAMES]
                [--evaluate_games EVALUATE_GAMES] [--environments ENVIRONMENTS]
                [--rounds ROUNDS] [--dir DIR] [--verbose]

Train PPO on Hex

options:
  -h, --help            show this help message and exit
  --size SIZE
  --seed SEED
  --train_games TRAIN_GAMES
  --evaluate_games EVALUATE_GAMES
  --environments ENVIRONMENTS
  --rounds ROUNDS
  --dir DIR
  --verbose
```

I got an good player in about 24 minutes on my Chromebook with the command line:

```
python train.py --size=5 --train_games 1000 --rounds=20
```

I play the game in a gui with `python gui.py` or in the terminal with `python
ui.py`.

Tkinter dies if I try to include Buttons or any other controls in the gui so
I plowed ahead creating my own buttons.

I'll add switch accessibility and hints later.

21 July 2025
