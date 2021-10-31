"""Trains a model on a dataset."""

from chemprop_fda.args import TrainArgs
from chemprop_fda.train import cross_validate
from chemprop_fda.utils import create_logger


if __name__ == '__main__':
    args = TrainArgs().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)