import argparse as ap
import os

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
    datasets = ["rach3", "pianoyt", "miditest"]
    dataset_dir_args = [[f"--{i}-dir"] for i in datasets]
    dataset_dir_kwargs = [{
        "default": os.environ.get(f"PPAN_{i.upper()}_DIR", None),
        "action": "store",
        "help": f"Path to {i} dataset",
        "required": False
    } for i in datasets]
    # Command line for the trainer
    parser_finetune = subparsers.add_parser(
        "finetune",
        help="Fine-tune the pre-trained network."
    )
    parser_finetune.add_argument(
        "-d", "--dataset-dir",
        action="store",
        default=os.environ.get("PPAN_PROCESSED_DATASET_DIR", None),
        help="Location of the processed dataset to train from.",
        required=False,
    )
    parser_finetune.add_argument(
        "-p", "--pretrained-checkpoint",
        action="store",
        default=os.environ.get("PPAN_PRETRAINED_MODEL_CHECKPOINT", None),
        help="Location of the pretrained model checkpoint to start "
             "fine-tuning from.",
        required=False,
    )
    parser_finetune.add_argument(
        "-o", "--output-dir",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_OUTPUT_DIR", None),
        help="Where to save the trained model, full path with filename.",
        required=False,
    )
    parser_finetune.add_argument(
        "-r", "--checkpoint-dir",
        action="store",
        help="Location of a training checkpoint when continuing training.",
        default=os.environ.get("PPAN_FINETUNE_MODEL_CHECKPOINT", None),
        required=False,
    )
    parser_finetune.add_argument(
        "--no-epochs",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_NO_EPOCHS", None),
        type=int,
        help="Number of epochs to train for",
        required=False
    )
    parser_finetune.add_argument(
        "--eval-every",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_EVAL_EVERY", None),
        type=int,
        help="How often to run the evaluation while training (in steps)",
        required=False
    )
    parser_finetune.add_argument(
        "--save-every",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_SAVE_EVERY", None),
        type=int,
        help="How often to save the model while training (in steps)",
        required=False
    )
    parser_finetune.add_argument(
        "--batch-size",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_BATCH_SIZE", None),
        type=int,
        help="Batch size to use when training",
        required=False
    )
    parser_finetune.add_argument(
        "--class-weights",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_CLASS_WEIGHTS", None),
        type=float,
        help="The weight of the positive class (1). For example, if you'd "
             "like to weigh it twice as much as a 0, put this value to 2.",
        required=False
    )
    parser_finetune.add_argument(
        "-lr", "--learning-rate",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_LR", None),
        type=float,
        help="The learning rate with which to train. Note that this is not "
             "the actual learning rate that is used, the actual one is "
             "calculated as lr * total_batch_size / 256. Where the total "
             "batch size is no_gpus * batch_size_per_gpu. This is according "
             "to the linear scaling rule: "
             "https://arxiv.org/abs/1706.02677",
        required=False
    )
    parser_finetune.add_argument(
        "--weight-decay",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_WEIGHT_DECAY", None),
        type=float,
        help="Weight decay to use with the AdamW optimizer.",
        required=False
    )
    parser_finetune.add_argument(
        "--warmup-ratio",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_WARMUP_RATIO", None),
        type=float,
        help="Percentage (0-1) of training samples to dedicate to the warmup "
             "phase of the lr scheduler.",
        required=False
    )
    parser_finetune.add_argument(
        "--scheduler-type",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_SCHEDULER_TYPE", None),
        type=str,
        help="What type of lr scheduler to use. Must be supported by the "
             "huggingface trainer. Default is cosine.",
        required=False
    )
    parser_finetune.add_argument(
        "--adam-beta1",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_ADAM_BETA1", None),
        type=float,
        help="Adam optimizer beta1 parameter.",
        required=False
    )
    parser_finetune.add_argument(
        "--adam-beta2",
        action="store",
        default=os.environ.get("PPAN_FINETUNE_ADAM_BETA2", None),
        type=float,
        help="Adam optimizer beta2 parameter.",
        required=False
    )
    parser_finetune.set_defaults(func=run_train)
    # Command line for pre-training
    parser_pretrain = subparsers.add_parser(
        "pretrain",
        help="Pre-Train the model on some data."
    )
    parser_pretrain.add_argument(
        "-d", "--dataset-dir",
        action="store",
        default=os.environ.get("PPAN_PROCESSED_DATASET_DIR"),
        help="Location of the processed dataset to train from.",
        required=False,
    )
    parser_pretrain.add_argument(
        "-o", "--output-dir",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_OUTPUT_DIR"),
        help="Where to save the trained model, full path with filename.",
        required=False,
    )
    parser_pretrain.add_argument(
        "-r", "--checkpoint-dir",
        action="store",
        help="Location of a training checkpoint when continuing training.",
        default=os.environ.get("PPAN_OUTPUT_DIR", None),
        required=False,
    )
    parser_pretrain.add_argument(
        "--no-epochs",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_NO_EPOCHS", None),
        type=int,
        help="Number of epochs to train for",
        required=False
    )
    parser_pretrain.add_argument(
        "--eval-every",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_EVAL_EVERY", None),
        type=int,
        help="How often to run the evaluation while training (in steps)",
        required=False
    )
    parser_pretrain.add_argument(
        "--save-every",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_SAVE_EVERY", None),
        type=int,
        help="How often to save the model while training (in steps)",
        required=False
    )
    parser_pretrain.add_argument(
        "--batch-size",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_BATCH_SIZE", None),
        type=int,
        help="Batch size to use when training",
        required=False
    )
    parser_pretrain.add_argument(
        "-lr", "--learning-rate",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_LR", None),
        type=float,
        help="The learning rate with which to train.",
        required=False
    )
    parser_pretrain.add_argument(
        "--weight-decay",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_WEIGHT_DECAY", None),
        type=float,
        help="Weight decay to use with the AdamW optimizer.",
        required=False
    )
    parser_pretrain.add_argument(
        "--warmup-ratio",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_WARMUP_RATIO", None),
        type=float,
        help="Percentage (0-1) of training samples to dedicate to the warmup "
             "phase of the lr scheduler.",
        required=False
    )
    parser_pretrain.add_argument(
        "--scheduler-type",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_SCHEDULER_TYPE", None),
        type=str,
        help="What type of lr scheduler to use. Must be supported by the "
             "huggingface trainer. Default is cosine.",
        required=False
    )
    parser_pretrain.add_argument(
        "--adam-beta1",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_ADAM_BETA1", None),
        type=float,
        help="Adam optimizer beta1 parameter.",
        required=False
    )
    parser_pretrain.add_argument(
        "--adam-beta2",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_ADAM_BETA2", None),
        type=float,
        help="Adam optimizer beta2 parameter.",
        required=False
    )
    parser_pretrain.add_argument(
        "--mask-ratio",
        action="store",
        default=os.environ.get("PPAN_PRETRAIN_MASK_RATIO", None),
        type=float,
        help="Adam optimizer beta2 parameter.",
        required=False
    )
    parser_pretrain.set_defaults(func=run_pretrain)

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
    parser_eval.set_defaults(func=run_evaluate)
    # Commnad line for dataset pre-pre-processing
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


def run_process(*args, **kwargs):
    from ppan.render_dataset import process
    process(*args, **kwargs)


def run_pretrain(*args, **kwargs):
    from ppan.pretrain import pretrain
    pretrain(*args, **kwargs)


def run_evaluate(*args, **kwargs):
    from ppan.evaluate_model import evaluate
    evaluate(*args, **kwargs)


def run_train(*args, **kwargs):
    from ppan.finetune import train
    train(*args, **kwargs)


if __name__ == "__main__":
    main()
