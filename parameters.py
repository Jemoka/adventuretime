import argparse

parser = argparse.ArgumentParser(prog='adventure')

# logistics
parser.add_argument("experiment", help="name for the experiment", type=str)
parser.add_argument('-v', '--verbose', action='count', default=0, help="log level")
parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
parser.add_argument("--warm_start", default=None, type=str, help="recover trainer from this path")

# intervals
parser.add_argument("--report_interval", default=64, type=int, help="save to wandb every this many steps")
parser.add_argument("--checkpoint_interval", default=256, type=int, help="checkpoint every this many steps")
parser.add_argument("--validation_interval", default=256, type=int, help="validate every this many steps")

# dataset
parser.add_argument("--out_dir", help="directory to save checkpoints and outputs", type=str, default="output")

# hyperparameters
parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=1)

