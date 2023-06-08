import argparse as ap

from ppan import train, evaluate


def main():
    parser = ap.ArgumentParser(
        prog="ppan",
        description="A model for analyzing piano playing videos and extracting "
                    "what notes are being played.",
        epilog="Coded with love by Uros Zivanovic"
    )
    subparsers = parser.add_subparsers(required=True)

    parser_train = subparsers.add_parser("train",
                                         help="Train the network on some "
                                              "data.")
    parser_train.add_argument(
        "-d", "--dataset_dir",
        nargs=1,
        action="store",
        type=str,
        help="Path to dataset to train on.",
        required=True
    )
    parser_train.add_argument(
        "-o", "--output",
        nargs=1,
        action="store",
        type=str,
        help="Where to save the trained model, full path with filename.",
        required=False,
        default=None
    )
    parser_train.add_argument(
        "-t", "--tensorboard",
        action="store_true",
        help="Whether to save tensorboard statistics in a ./runs dir.",
        default=None,
        required=False
    )
    parser_train.set_defaults(func=train.main)

    parser_eval = subparsers.add_parser("eval",
                                        help="Evaluate the model on a "
                                             "testset.")
    parser_eval.add_argument(
        "-d", "--dataset_dir",
        nargs=1,
        action="store",
        type=str,
        help="Path to dataset to train on.",
        required=True
    )
    parser_eval.add_argument(
        "-m", "--model_dir",
        nargs=1,
        action="store",
        type=str,
        help="Path to the saved model.",
        required=True
    )
    parser_eval.set_defaults(func=evaluate.main)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
