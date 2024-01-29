import argparse as ap

from ppan.train import main as train
from ppan.evaluate_model import main as evaluate_model


def main():
    parser = ap.ArgumentParser(
        prog="ppan",
        description="A model for analyzing piano playing videos and "
                    "extracting what notes are being played.",
        epilog="Coded with love by Uros Zivanovic"
    )
    subparsers = parser.add_subparsers(required=True)

    dataset_dir_args = ["-d", "--dataset-dir"]
    dataset_dir_kwargs = {
        "nargs": "*",
        "action": "store",
        "help": "Path to rach3 and piano_yt datasets",
        "required": True
    }

    parser_train = subparsers.add_parser("train",
                                         help="Train the network on some "
                                              "data.")
    parser_train.add_argument(
        *dataset_dir_args,
        **dataset_dir_kwargs
    )
    parser_train.add_argument(
        "-o", "--output-dir",
        nargs=1,
        action="store",
        type=str,
        help="Where to save the trained model, full path with filename.",
        required=True,
    )
    parser_train.add_argument(
        "-r", "--checkpoint-dir",
        nargs=1,
        action="store",
        type=str,
        help="Location of a training checkpoint when continuing training.",
        required=False,
        default=None
    )
    parser_train.set_defaults(func=train)

    parser_eval = subparsers.add_parser("evaluate",
                                        help="Run inference using a trained "
                                             "model and pickle the outputs.")
    parser_eval.add_argument(
        *dataset_dir_args,
        **dataset_dir_kwargs
    )
    parser_eval.add_argument(
        "-m", "--model-checkpoint",
        nargs=1,
        action="store",
        type=str,
        help="Path to the saved model.",
        required=True
    )
    parser_eval.add_argument(
        "-o", "--output",
        nargs=1,
        action="store",
        type=str,
        help="Where to save the generated predictions, including filename "
             "with .pkl at the end.",
        required=True,
    )
    parser_eval.set_defaults(func=evaluate_model)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
