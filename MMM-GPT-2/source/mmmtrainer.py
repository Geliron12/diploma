import os
import numpy as np
import random
import collections
import torch
from typing import Dict
from torch.utils.data.dataset import Dataset
from tokenizers import Tokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from source import logging

logger = logging.create_logger("mmmtrainer")


class MMMTrainer:

    def __init__(self,
        tokenizer_path="",
        dataset_train_files=[],
        dataset_validate_files=[],
        pad_length=768,
        shuffle_buffer_size=10000,
        batch_size=4,
        epochs=10,
        n_head=8,
        n_layer=6,
        n_embd=512,
        n_positions=1024,
        n_ctx=1024
        ):

        assert pad_length <= n_positions

        self.tokenizer_path = tokenizer_path
        self.dataset_train_files = dataset_train_files
        self.dataset_validate_files = dataset_validate_files
        self.pad_length = pad_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.n_ctx = n_ctx

    def train(self, output_path, simulate=False):

        if torch.cuda.is_available():
            logger.info("Found a GPU.")
        else:
            logger.warning("Did not find a GPU.")

        # создание токенайзера
        if not os.path.exists(self.tokenizer_path):
            raise Exception(f"No tokenizer found at {self.tokenizer_path}")
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
        pretrained_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # модель
        model_config = GPT2Config(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=tokenizer.token_to_id("[PAD]"),
            n_head=self.n_head,
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_positions=self.n_positions,
            n_ctx=self.n_ctx
        )
        logger.info(model_config)
        model = GPT2LMHeadModel(model_config)

        # подготовка тренировочного набора
        print("Preparing training dataset...")
        dataset_train = TokenSequenceDataset(
            tokenizer=pretrained_tokenizer,
            dataset_paths=self.dataset_train_files,
            block_size=self.pad_length,
            simulate=simulate
        )
        logger.info("Training dataset prepared.")

        # валидационный набор
        print("Preparing validate dataset...")
        dataset_valid = TokenSequenceDataset(
            tokenizer=pretrained_tokenizer,
            dataset_paths=self.dataset_validate_files,
            block_size=self.pad_length,
            simulate=simulate
        )
        logger.info("Validation dataset prepared.")

        data_collator = DataCollatorWithPadding(
            tokenizer=pretrained_tokenizer,
            padding="max_length",
            max_length=self.pad_length
        )

        print("Creating trainer...")
        training_args = TrainingArguments(
            output_dir=os.path.join(output_path),
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            num_train_epochs=self.epochs,
            per_gpu_train_batch_size=self.batch_size,
            save_steps=1_000,
            save_total_limit=2,
            prediction_loss_only=False,
            logging_strategy="steps",
            logging_dir=os.path.join(output_path, "logs"),
            load_best_model_at_end=True,
            save_strategy="steps"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid
        )

        # обучение
        logger.info("Training the model...")
        trainer.train()

        # сохранение модели
        model_path = os.path.join(output_path, "best_model")
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}.")


class TokenSequenceDataset(Dataset):

    def __init__(self, tokenizer, dataset_paths, block_size, simulate=False):

        pad_token_id = tokenizer.encode("[PAD]")[0]
        unk_token_id = tokenizer.encode("[UNK]")[0]

        lines = []
        for dataset_path in dataset_paths:
            assert os.path.isfile(dataset_path), f"Input file path {dataset_path} not found"
            lines += open(dataset_path, "r").readlines()

        if simulate:
            random.shuffle(lines)
            lines = lines[:10]

        # создаем тренировочные семплы
        self.examples = []
        unknown_tokens_set = []
        unknown_tokens = []
        tokens_count = 0
        unknown_token_lines_count = 0
        too_long_lines_count = 0
        encoded_lengths = []
        for line in tqdm(lines):

            line = line.strip()
            if line == "":
                continue

            encoded_line = tokenizer.encode(line)
            encoded_lengths += [len(encoded_line)]
            tokens_count += len(encoded_line)

            # неизвестные токены
            if unk_token_id in encoded_line:
                index = encoded_line.index(unk_token_id)
                token = tokenizer.decode(encoded_line[index])
                token = line.split()[index]
                if token not in unknown_tokens_set:
                    unknown_tokens_set += [token]
                unknown_tokens += [token]
                unknown_token_lines_count += 1
                continue

            if len(encoded_line) > block_size:
                too_long_lines_count += 1
                continue

            tensor = np.full((block_size,), pad_token_id, dtype=np.long)
            tensor[:len(encoded_line)] = encoded_line
            assert len(tensor) == block_size

            self.examples += [{
                "input_ids": torch.tensor(tensor, dtype=torch.long),
                "labels": torch.tensor(tensor, dtype=torch.long)
            }]

        # Статистика по данным
        logger.info(f"Minimum sequence length before padding: {np.min(encoded_lengths)}")
        logger.info(f"Mean sequence length before padding:    {np.mean(encoded_lengths)}")
        logger.info(f"STD sequence length before padding:     {np.std(encoded_lengths)}")
        logger.info(f"Maximum sequence length before padding: {np.max(encoded_lengths)}")
        logger.info(f"Number of tokens: {tokens_count}")
        for key, value in collections.Counter(unknown_tokens).most_common(1000):
            logger.info(f"Unknown token {key} count {value}, {100 * value / len(unknown_tokens):.2f}% of all unknown tokens.")
        logger.info(f"Lines with unknown tokens {unknown_token_lines_count}/{len(lines)}, {100 * unknown_token_lines_count / len(lines):.2f}%.")
        logger.info(f"Too long lines {too_long_lines_count}/{len(lines)}, {100 * too_long_lines_count / len(lines):.2f}%.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class JSBTrack(MMMTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)