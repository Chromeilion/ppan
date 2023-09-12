import random
from pathlib import Path

import datasets
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    BertForPreTraining,
    Trainer,
    TrainingArguments,
    BertConfig,
    DataCollatorForLanguageModeling,
)

from ppan.config import seed
from ppan.types import PathLike


def train_tokenizer(iterator):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=150,
        special_tokens=special_tokens
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    return tokenizer


def train_and_save_tokenizer(dataset: PathLike, tokenizer_path: Path):
    iterator = dataset_iterator(dataset)

    if not tokenizer_path.exists():
        tokenizer_path.mkdir()

    tokenizer = train_tokenizer(iterator=iterator)
    tokenizer_wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer_wrapped.save_pretrained(save_directory=tokenizer_path,
                                      push_to_hub=False)


def train_and_save_decoder(train, test, training_arguments: TrainingArguments,
                           tokenizer):
    bert_config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=4,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.2,
        return_tensors="pt"
    )
    model = BertForPreTraining(config=bert_config).train()
    trainer = Trainer(
        model=model,
        args=training_arguments,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train,
        eval_dataset=test
    )
    trainer.train()


def prepare_dataset(dataset, tokenizer):
    num_proc = 15
    dataset = dataset.filter(lambda sample: sample['sentence'] is not None,
                             num_proc=num_proc)
    processor = lambda sample, idx: process_sentance(sample, tokenizer,
                                                     idx, dataset)
    dataset = dataset.map(processor, with_indices=True, num_proc=num_proc)
    dataset = dataset.remove_columns(['sentence', 'file_no'])
    dataset = dataset.filter(lambda example, idx: idx % 2 == 0,
                             with_indices=True,
                             num_proc=num_proc)
    return dataset


def process_sentance(sentence, tokenizer, idx, dataset):
    dataset_len = len(dataset)
    label = None
    next_idx = idx+1
    if random.random() < 0.49 or next_idx >= dataset_len:
        new_idx = next_idx
        label = 0
        while new_idx == next_idx:
            new_idx = random.randint(0, dataset_len-1)
        next_idx = new_idx
    current_file = sentence['file_no']
    next_file = dataset[next_idx]['file_no']
    if label != 0:
        if current_file != next_file:
            label = 0
        else:
            label = 1
    batch_encoding = tokenizer(
        sentence['sentence'],
        dataset[next_idx]['sentence'],
        return_special_tokens_mask=True,
        truncation=True,
        max_length=40,
    )
    batch_encoding.data["next_sentence_label"] = label
    return batch_encoding


def main(dataset_dir: PathLike, output_dir: PathLike, tokenizer: bool,
         decoder_model: bool, *_, **__):
    if not tokenizer and not decoder_model:
        return

    output_dir = Path(output_dir[0])
    dataset = datasets.load_dataset(dataset_dir[0])
    train = dataset["train"]
    test = dataset["test"]
    tokenizer_path = output_dir.joinpath("tokenizer")
    if tokenizer:
        train_and_save_tokenizer(train, tokenizer_path)

    if decoder_model:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            tokenizer_path)
        train = prepare_dataset(train, tokenizer)
        test = prepare_dataset(test, tokenizer)
        training_arguments = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="steps",
            eval_steps=1000,
            logging_steps=1000,
            num_train_epochs=40,
            learning_rate=1e-4,
            do_train=True,
            do_eval=True,
            lr_scheduler_type="cosine",
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.01,
            warmup_ratio=0.3,
            logging_nan_inf_filter=True,
            seed=seed,
            optim="adamw_torch",
            auto_find_batch_size=True,
            save_steps=5000
        )
        train_and_save_decoder(train, test, training_arguments,
                               tokenizer)


def dataset_iterator(dataset):
    for i in dataset:
        if i["sentence"] is None:
            continue
        yield i["sentence"]
