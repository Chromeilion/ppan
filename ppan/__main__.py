import argparse as ap

from ppan import train, evaluate_model, finetune
from ppan.preprocessing import preprocess, pretrain, playing_detector


def main():
    parser = ap.ArgumentParser(
        prog="ppan",
        description="A model for analyzing piano playing videos and extracting "
                    "what notes are being played.",
        epilog="Coded with love by Uros Zivanovic"
    )
    subparsers = parser.add_subparsers(required=True)

    dataset_dir_args = ["-d", "--dataset-dir"]
    dataset_dir_kwargs = {
        "nargs": 1,
        "action": "store",
        "type": str,
        "help": "Path to dataset",
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
        "-t", "--tokenizer-dir",
        nargs=1,
        action="store",
        type=str,
        help="Location of pretrained tokenizer folder.",
        required=True
    )
    parser_train.add_argument(
        "-c", "--decoder-checkpoint",
        nargs=1,
        action="store",
        type=str,
        help="Location of pretrained decoder",
        required=True
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
    parser_train.set_defaults(func=train.main)

    parser_train_det = subparsers.add_parser("train_detector",
                                             help="Train the playing "
                                                  "detector network on some "
                                                  "data.")
    parser_train_det.add_argument(
        *dataset_dir_args,
        **dataset_dir_kwargs
    )
    parser_train_det.add_argument(
        "-o", "--output-dir",
        nargs=1,
        action="store",
        type=str,
        help="Where to save the trained model, full path with filename.",
        required=True,
    )
    parser_train_det.add_argument(
        "-r", "--checkpoint-dir",
        nargs=1,
        action="store",
        type=str,
        help="Location of a training checkpoint when continuing training.",
        required=False,
        default=None
    )
    parser_train_det.set_defaults(func=playing_detector.main)
    parser_finetune = subparsers.add_parser("finetune",
                                            help="Fine tune the model on an "
                                                 "external dataset such as "
                                                 "ytmidi.")
    parser_finetune.add_argument(
        *dataset_dir_args,
        **dataset_dir_kwargs
    )
    parser_finetune.add_argument(
        "-t", "--tokenizer-dir",
        nargs=1,
        action="store",
        type=str,
        help="Location of pretrained tokenizer folder.",
        required=True
    )
    parser_finetune.add_argument(
        "-o", "--output-dir",
        nargs=1,
        action="store",
        type=str,
        help="Where to save the trained models, full path with filename.",
        required=True,
    )
    parser_finetune.add_argument(
        "--dataset-type",
        nargs=1,
        action="store",
        type=str,
        help="Type of dataset.",
        required=False,
        default=None
    )
    parser_finetune.add_argument(
        "--pretrained-note-model",
        nargs=1,
        action="store",
        type=str,
        help="Path to pretrained note model",
        required=False,
        default=None
    )
    parser_finetune.add_argument(
        "--pretrained-det-model",
        nargs=1,
        action="store",
        type=str,
        help="Path to pretrained playing detection model",
        required=False,
        default=None
    )
    parser_finetune.set_defaults(func=finetune.main)
    parser_eval = subparsers.add_parser("evaluate",
                                        help="Run inference using a trained "
                                             "model and pickle the outputs.")
    parser_eval.add_argument(
        *dataset_dir_args,
        **dataset_dir_kwargs
    )
    parser_eval.add_argument(
        "--youtubemidi",
        nargs=1,
        action="store",
        type=str,
        help="Path to youtube dataset",
        required=False,
        default=None
    )
    parser_eval.add_argument(
        "--miditest",
        nargs=1,
        action="store",
        type=str,
        help="Path to MIDI test dataset",
        required=False,
        default=None
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
        "--detector-checkpoint",
        nargs=1,
        action="store",
        type=str,
        help="Path to the saved playing detector model.",
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
    parser_eval.add_argument(
        "-t", "--tokenizer-dir",
        nargs=1,
        action="store",
        type=str,
        help="Location of pretrained tokenizer folder.",
        required=True
    )
    parser_eval.set_defaults(func=evaluate_model.main)

    parser_preprocess = subparsers.add_parser(
        "preprocess",
        help="Perform preprocessing steps before training can be done.")
    parser_preprocess.add_argument(
        *dataset_dir_args,
        **dataset_dir_kwargs
    )
    parser_preprocess.add_argument(
        "-o", "--output-dir",
        nargs=1,
        action="store",
        type=str,
        help="Where to output preprocessed files",
        required=True
    )
    parser_preprocess.add_argument(
        "-n", "--dataset_name",
        required=False,
        nargs=1,
        action="store",
        type=str,
        help="When using splits, the names are autogenerated as train/test. "
             "If not splitting the name must be given manually."
    )
    parser_preprocess.set_defaults(func=preprocess.main)

    parser_pretrain = subparsers.add_parser(
        "pretrain",
        help="Perform pretraining steps to prepare the tokenizer and decoder.")
    parser_pretrain.add_argument(
        *dataset_dir_args,
        **dataset_dir_kwargs
    )
    parser_pretrain.add_argument(
        "-k", "--tokenizer",
        action="store_true",
        help="Whether to train the tokenizer. If not training tokenizer then "
             "output-dir is searched for an already trained one."
    )
    parser_pretrain.add_argument(
        "-m", "--decoder-model",
        action="store_true",
        help="Whether to train the decoder."
    )
    parser_pretrain.add_argument(
        "-o", "--output-dir",
        nargs=1,
        action="store",
        type=str,
        help="Where to save the trained tokenizer and decoder.",
        required=True
    )
    parser_pretrain.set_defaults(func=pretrain.main)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
