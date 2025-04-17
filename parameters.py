import argparse

parser = argparse.ArgumentParser(prog='adventure')

# logistics
parser.add_argument("experiment", help="name for the experiment", type=str)
parser.add_argument('-v', '--verbose', action='count', default=0, help="log level")
parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
parser.add_argument("--warm_start", default=None, type=str, help="recover trainer from this path")

# intervals
parser.add_argument("--report_interval", default=64, type=int, help="save to wandb every this many steps")
parser.add_argument("--plot_interval", default=256, type=int, help="checkpoint every this many steps")
parser.add_argument("--checkpoint_interval", default=256, type=int, help="checkpoint every this many steps")
parser.add_argument("--validation_interval", default=256, type=int, help="validate every this many steps")

# dataset
parser.add_argument("--out_dir", help="directory to save checkpoints and outputs", type=str, default="output")

# hyperparameters
parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=1)
## optimizer configuration
parser.add_argument("--weight_decay", type=float, default=1e-1, help="AdamW weight decay")
parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 parameter") 
parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2 parameter")

# GPT model construction arguments
parser.add_argument("--block_size", help="context length", type=int, default=1024)
parser.add_argument("--vocab_size", help="vocabulary size", type=int, default=50304)
parser.add_argument("--n_layer", help="number of layers", type=int, default=12)
parser.add_argument("--n_head", help="number of attention heads", type=int, default=12)
parser.add_argument("--n_embd", help="embedding size", type=int, default=768)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.0)
parser.add_argument("--no_bias", help="do not use bias in linear layers", action="store_false", default=True)

