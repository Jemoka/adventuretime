from loguru import logger

from trainer import Trainer
from parameters import parser

@logger.catch
def execute(args):
    if args.warm_start != None:
        # by default, the from_pretrained function disables
        # whatever wandb settings was there b/c we usually
        # use this to load an existing model, but when we are
        # actually training, we want to actually enable it
        trainer = Trainer.from_pretrained(args.warm_start,
                                          disable_wandb=False)
    else:
        trainer = Trainer(args)

    # <<<<<<< do something with trainer <<<<<<<
    #
    # trainer.train()
    #
    # >>>>>>> do something with trainer >>>>>>>

    raise NotImplementedError("ideally, this does something")

def configure(experiment, **kwargs):
    """configure a run from arguments

    Arguments
    ----------
        experiment : str
                experiment name
        kwargs : dict
                arguments to configure the run

    Returns
    -------
        SimpleNamespace
                configuration object
    """

    # listcomp grossery to parse input string into arguments that's
    # readable by argparse

    try:
        return parser.parse_args(([str(experiment)]+
        [j for k,v in kwargs.items() for j in ([f"--{k}", str(v)]
        if not isinstance(v, bool) else [f"--{k}"])]))
    except SystemExit as e:
        logger.error("unrecognized arguments found in configure: {}", kwargs)
        return None




