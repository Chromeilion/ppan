import argparse as ap
import os

from dotenv import load_dotenv

from ppan.evaluate_model import evaluate
from ppan.train import train as train


def main():
    load_dotenv()
    parser = ap.ArgumentParser(
        prog="PPAn",
        description="A model for analyzing piano playing videos and "
                    "extracting what notes are being played.",
        epilog="Coded with love by Uros Zivanovic"
    )
    subparsers = parser.add_subparsers(required=True)

    datasets = ["rach3", "pianoyt", "miditest"]
    dataset_dir_args = [[f"--{i}-dir"] for i in datasets]
    dataset_dir_kwargs = [{
        "default": os.environ.get(f"PPAN_{i.upper()}_DIR"),
        "action": "store",
        "help": f"Path to {i} dataset",
        "required": False
    } for i in datasets]
    parser_train = subparsers.add_parser(
        "train",
        help="Train the network on some data."
    )
    [parser_train.add_argument(
        *i,
        **j
    ) for i, j in zip(dataset_dir_args, dataset_dir_kwargs)]
    parser_train.add_argument(
        "-o", "--output-dir",
        action="store",
        default=os.environ.get("PPAN_OUTPUT_DIR"),
        help="Where to save the trained model, full path with filename.",
        required=False,
    )
    parser_train.add_argument(
        "-r", "--checkpoint-dir",
        action="store",
        help="Location of a training checkpoint when continuing training.",
        default=os.environ.get("PPAN_MODEL_CHECKPOINT", None),
        required=False,
    )
    parser_train.add_argument(
        "--no-epochs",
        action="store",
        default=os.environ.get("PPAN_NO_EPOCHS", None),
        type=int,
        help="Number of epochs to train for",
        required=False
    )
    parser_train.add_argument(
        "--eval-every",
        action="store",
        default=os.environ.get("PPAN_EVAL_EVERY", None),
        type=int,
        help="How often to run the evaluation while training (in steps)",
        required=False
    )
    parser_train.add_argument(
        "--save-every",
        action="store",
        default=os.environ.get("PPAN_SAVE_EVERY", None),
        type=int,
        help="How often to save the model while training (in steps)",
        required=False
    )
    parser_train.add_argument(
        "--batch-size",
        action="store",
        default=os.environ.get("PPAN_BATCH_SIZE", None),
        type=int,
        help="Batch size to use when training",
        required=False
    )
    parser_train.add_argument(
        "--max-iters-per-epoch-test",
        action="store",
        default=os.environ.get("PPAN_MAX_ITERS_PER_EPOCH_TEST", None),
        type=int,
        help="Maximum amount of iterations per test epoch, for when you "
             "dont want to iterate through the entire test dataset every "
             "time validation is run.",
        required=False
    )
    parser_train.add_argument(
        "--max-iters-per-epoch-train",
        action="store",
        default=os.environ.get("PPAN_MAX_ITERS_PER_EPOCH_TRAIN", None),
        type=int,
        help="Maximum amount of iterations per train epoch, for when you dont "
             "want to iterate through the entire train dataset every epoch.",
        required=False
    )
    parser_train.add_argument(
        "--class-weights",
        action="store",
        default=os.environ.get("PPAN_CLASS_WEIGHTS", None),
        type=float,
        help="The weight of the positive class (1). For example, if you'd "
             "like to weigh it twice as much as a 0, put this value to 2.",
        required=False
    )
    parser_train.add_argument(
        "-lr", "--learning-rate",
        action="store",
        default=os.environ.get("PPAN_LR", None),
        type=float,
        help="The learning rate with which to train. Note that this is not "
             "the actual learning rate that is used, the actual one is "
             "calculated as lr * total_batch_size / 256. Where the total "
             "batch size is no_gpus * batch_size_per_gpu. This is according "
             "to the linear scaling rule: "
             "https://arxiv.org/abs/1706.02677",
        required=False
    )
    parser_train.add_argument(
        "--weight-decay",
        action="store",
        default=os.environ.get("PPAN_WEIGHT_DECAY", None),
        type=float,
        help="Weight decay to use with the AdamW optimizer.",
        required=False
    )
    parser_train.add_argument(
        "--warmup-ratio",
        action="store",
        default=os.environ.get("PPAN_WARMUP_RATIO", None),
        type=float,
        help="Percentage (0-1) of training samples to dedicate to the warmup "
             "phase of the lr scheduler.",
        required=False
    )
    parser_train.add_argument(
        "--scheduler-type",
        action="store",
        default=os.environ.get("PPAN_SCHEDULER_TYPE", None),
        type=str,
        help="What type of lr scheduler to use. Must be supported by the "
             "huggingface trainer. Default is cosine.",
        required=False
    )
    parser_train.add_argument(
        "--adam-beta1",
        action="store",
        default=os.environ.get("PPAN_ADAM_BETA1", None),
        type=float,
        help="Adam optimizer beta1 parameter.",
        required=False
    )
    parser_train.add_argument(
        "--adam-beta2",
        action="store",
        default=os.environ.get("PPAN_ADAM_BETA2", None),
        type=float,
        help="Adam optimizer beta2 parameter.",
        required=False
    )
    parser_train.set_defaults(func=train)

    parser_eval = subparsers.add_parser(
        "evaluate",
        help="Run inference using a trained model, calculate MIR statistics, "
             "and save the outputs as a MIDI file."
    )
    [parser_eval.add_argument(
        *i,
        **j
    ) for i, j in zip(dataset_dir_args, dataset_dir_kwargs)]
    parser_eval.add_argument(
        "-m", "--model-checkpoint",
        action="store",
        default=os.environ.get("PPAN_MODEL_CHECKPOINT", None),
        help="Path to the trained model.",
        required=False
    )
    parser_eval.add_argument(
        "-o", "--preds-output",
        action="store",
        default=os.environ.get("PPAN_EVAL_PREDS_OUTPUT", None),
        help="The pickle file where model predictions should be saved.",
        required=False,
    )
    parser_eval.add_argument(
        "-om", "--midi-output",
        action="store",
        default=os.environ.get("PPAN_EVAL_MIDI_OUTPUT", None),
        help="The folder where predicted MIDI files should be saved.",
        required=False,
    )
    parser_eval.add_argument(
        "--batch-size",
        action="store",
        default=os.environ.get("PPAN_EVAL_BATCH_SIZE", None),
        help="The folder where predicted MIDI files should be saved.",
        required=False,
    )
    parser_eval.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
