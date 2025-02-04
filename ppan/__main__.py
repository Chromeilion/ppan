import argparse as ap
import os
import json

from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = ap.ArgumentParser(
        prog="PPAn",
        description="A model for analyzing piano playing videos and "
                    "extracting what notes are being played.",
        epilog="Coded with love by Uros Zivanovic"
    )
    subparsers = parser.add_subparsers(required=True)

    # Reusable args for the datasets
    datasets = ["rach3_s", "rach3_x", "pianoyt", "miditest"]
    dataset_dir_args = [[f"--{i}-dir"] for i in datasets]
    dataset_dir_kwargs = [{
        "default": os.environ.get(f"PPAN_{i.upper()}_DIR", None),
        "action": "store",
        "help": f"Path to {i} dataset",
        "required": False
    } for i in datasets]
    # Command line for the trainer
    parser_train = subparsers.add_parser(
        "train",
        help="Train the model on a dataset."
    )
    parser_train.add_argument(
        "-d", "--dataset-dir",
        action="store",
        default=os.environ.get("PPAN_PROCESSED_DATASET_DIR", None),
        help="Location of the processed dataset to train from.",
        required=False,
    )
    parser_train.add_argument(
        "-p", "--pretrained-checkpoint",
        action="store",
        default=os.environ.get("PPAN_PRETRAINED_MODEL_CHECKPOINT", None),
        help="Location of the pretrained PPAN model checkpoint to start "
             "training from.",
        required=False,
    )
    parser_train.add_argument(
        "-o", "--output-dir",
        action="store",
        default=os.environ.get("PPAN_TRAIN_OUTPUT_DIR", None),
        help="Where to save the trained model, full path with filename.",
        required=False,
    )
    parser_train.add_argument(
        "-r", "--checkpoint-dir",
        action="store",
        help="Location of a training checkpoint when continuing training.",
        default=os.environ.get("PPAN_TRAIN_MODEL_CHECKPOINT", None),
        required=False,
    )
    parser_train.add_argument(
        "--no-epochs",
        action="store",
        default=os.environ.get("PPAN_TRAIN_NO_EPOCHS", None),
        type=int,
        help="Number of epochs to train for",
        required=False
    )
    parser_train.add_argument(
        "--eval-every",
        action="store",
        default=os.environ.get("PPAN_TRAIN_EVAL_EVERY", None),
        type=int,
        help="How often to run the evaluation while training (in steps)",
        required=False
    )
    parser_train.add_argument(
        "--save-every",
        action="store",
        default=os.environ.get("PPAN_TRAIN_SAVE_EVERY", None),
        type=int,
        help="How often to save the model while training (in steps)",
        required=False
    )
    parser_train.add_argument(
        "--batch-size",
        action="store",
        default=os.environ.get("PPAN_TRAIN_BATCH_SIZE", None),
        type=int,
        help="Batch size to use when training",
        required=False
    )
    parser_train.add_argument(
        "-lr", "--learning-rate",
        action="store",
        default=os.environ.get("PPAN_TRAIN_LR", None),
        type=float,
        help="The learning rate with which to train.",
        required=False
    )
    parser_train.add_argument(
        "-w", "--class-weights",
        action="store",
        default=os.environ.get("PPAN_TRAIN_CLASS_WEIGHTS", None),
        type=json.loads,
        help="Custom class weights to be applied to the loss",
        required=False
    )
    parser_train.add_argument(
        "--weight-decay",
        action="store",
        default=os.environ.get("PPAN_TRAIN_WEIGHT_DECAY", None),
        type=float,
        help="Weight decay to use with the optimizer.",
        required=False
    )
    parser_train.add_argument(
        "--warmup-ratio",
        action="store",
        default=os.environ.get("PPAN_TRAIN_WARMUP_RATIO", None),
        type=float,
        help="Percentage (0-1) of all training steps to dedicate to the warmup "
             "phase of the lr scheduler.",
        required=False
    )
    parser_train.add_argument(
        "--scheduler-type",
        action="store",
        default=os.environ.get("PPAN_TRAIN_SCHEDULER_TYPE", None),
        type=str,
        help="What type of lr scheduler to use. Must be supported by the "
             "huggingface trainer. Default is cosine.",
        required=False
    )
    parser_train.add_argument(
        "--adam-beta1",
        action="store",
        default=os.environ.get("PPAN_TRAIN_ADAM_BETA1", None),
        type=float,
        help="Adam optimizer beta1 parameter.",
        required=False
    )
    parser_train.add_argument(
        "--adam-beta2",
        action="store",
        default=os.environ.get("PPAN_TRAIN_ADAM_BETA2", None),
        type=float,
        help="Adam optimizer beta2 parameter.",
        required=False
    )
    parser_train.add_argument(
        "-sj", "--spatial-jitter",
        action="store",
        help="Whether to use the spatial jitter augmentation",
        default=get_boolian_env("PPAN_TRAIN_SPATIAL_JITTER"),
        type=str_to_bool,
    )
    parser_train.add_argument(
        "-cg", "--color-jitter",
        action="store",
        help="Whether to use the color jitter augmentation",
        default=get_boolian_env("PPAN_TRAIN_COLOR_JITTER"),
        type=str_to_bool,
    )
    parser_train.add_argument(
        "-oo", "--onsets-only",
        action="store",
        help="Whether to train only on onset predictions",
        default=get_boolian_env("PPAN_TRAIN_ONSETS_ONLY"),
        type=str_to_bool
    )
    parser_train.add_argument(
        "-fo", "--frames-only",
        action="store",
        help="Whether to train only on frame predictions",
        default=get_boolian_env("PPAN_TRAIN_FRAMES_ONLY"),
        type=str_to_bool
    )
    parser_train.add_argument(
        "-rr", "--rand-rotate",
        action="store",
        help="Whether to use random rotation augmentation",
        default=get_boolian_env("PPAN_TRAIN_RAND_ROTATE"),
        type=str_to_bool,
    )
    parser_train.add_argument(
        "-dr", "--dropout",
        action="store",
        help="Amount of dropout to use",
        default=os.environ.get("PPAN_TRAIN_DROPOUT", None),
        type=float
    )
    parser_train.add_argument(
        "-dp", "--drop-path",
        action="store",
        help="Stochastic dropout parameter",
        default=os.environ.get("PPAN_TRAIN_DROP_PATH", None),
        type=float
    )
    parser_train.add_argument(
        "-g", "--grayscale",
        action="store",
        help="Whether to train on grayscale videos",
        default=get_boolian_env("PPAN_TRAIN_GRAYSCALE"),
        type=str_to_bool
    )
    parser_train.add_argument(
        "-re", "--rand-erase",
        action="store",
        help="Whether to use random erasing augmentation",
        default=get_boolian_env("PPAN_TRAIN_RAND_ERASE"),
        type=str_to_bool
    )
    parser_train.add_argument(
        "-gn", "--gaussian-noise",
        action="store",
        help="Whether to use gaussian noise augmentation",
        default=get_boolian_env("PPAN_TRAIN_GAUSSIAN_NOISE"),
        type=str_to_bool,
    )
    parser_train.add_argument(
        "-lsf", "--label-smoothing-conf-frame",
        action="store",
        help="Label smoothing confidence for frame predictions",
        default=os.environ.get("PPAN_TRAIN_LABEL_SMOOTHING_CONF_FRAME", None),
    )
    parser_train.add_argument(
        "-lso", "--label-smoothing-conf-onset",
        action="store",
        help="Label smoothing confidence for onset predictions",
        default=os.environ.get("PPAN_TRAIN_LABEL_SMOOTHING_CONF_ONSET", None),
    )
    parser_train.add_argument(
        "--model-architecture",
        action="store",
        help="What model architecture to use (vit_s, vit_b, cnn)",
        default=os.environ.get("PPAN_TRAIN_MODEL_ARCHITECTURE", None),
    )
    parser_train.add_argument(
        "--window-size",
        action="store",
        help="Number of frames to give to the model",
        default=os.environ.get("PPAN_TRAIN_WINDOW_SIZE", None),
        type=int,
    )
    parser_train.add_argument(
        "-m", "--momentum",
        action="store",
        help="Momentum parameter for SGD",
        default=os.environ.get("PPAN_TRAIN_MOMENTUM", None),
    )
    parser_train.add_argument(
        "-opt", "--optimizer",
        action="store",
        help="Optimizer to use. Either 'adam' or 'sgd'",
        default=os.environ.get("PPAN_TRAIN_OPTIMIZER", None),
    )
    parser_train.add_argument(
        "--image-size",
        action="store",
        help="H/W resolution of the model input frames as a tuple of ints",
        default=os.environ.get("PPAN_TRAIN_IMAGE_SIZE", None),
        type=json.loads
    )
    parser_train.set_defaults(func=run_train)
    # Command line for evaluation
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
        "-g", "--greyscale",
        action="store",
        default=get_boolian_env(os.environ.get("PPAN_EVAL_GREYSCALE", None)),
        help="Whether or not the model is greyscale.",
        type=str_to_bool,
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
        help="Number of frames to process at once.",
        required=False,
    )
    parser_eval.set_defaults(func=run_evaluate)
    # Command line for dataset pre-processing
    parser_process = subparsers.add_parser(
        "process",
        help="Process the dataset into a more convenient format."
    )
    [parser_process.add_argument(
        *i,
        **j
    ) for i, j in zip(dataset_dir_args, dataset_dir_kwargs)]
    parser_process.add_argument(
        "-o", "--output-dir",
        action="store",
        default=os.environ.get("PPAN_RENDER_OUTPUT_DIR", None),
        help="Directory to store the processed dataset",
        required=False,
    )
    parser_process.set_defaults(func=run_process)

    args = parser.parse_args()
    args.func(**vars(args))


def get_boolian_env(key):
    return str_to_bool(os.environ.get(key, "False"))

def str_to_bool(input_str):
    if input_str is None:
        return None
    return input_str.lower() in ("yes", "true", "t", "1")

# To speed up the CLI we only import the rest of the package once all
# args have been processed.

def run_process(*args, **kwargs):
    from ppan.render_dataset import process
    process(*args, **kwargs)


def run_evaluate(*args, **kwargs):
    from ppan.evaluate import evaluate
    evaluate(*args, **kwargs)


def run_train(*args, **kwargs):
    from ppan.train import train
    train(*args, **kwargs)


if __name__ == "__main__":
    main()
