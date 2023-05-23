import argparse as ap

from ppan import train

parser = ap.ArgumentParser(
    prog="ppan",
    description="A model for analyzing piano playing videos and extracting "
                "what notes are being played.",
    epilog="Coded with love by Uros Zivanovic"
)
subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser("train", help="Train the network on some "
                                                   "data.")
parser_train.add_argument(
    "-d", "--dataset_dir",
    nargs=1,
    action="store",
    type=str,
    help="Path to dataset to train on."
)
parser_train.add_argument(
    "-o", "--output",
    nargs=1,
    action="store",
    type=str,
    help="Where to save the trained model, full path with filename."
)
parser_train.set_defaults(func=train.main)

args = parser_train.parse_args()
args.func(args)
