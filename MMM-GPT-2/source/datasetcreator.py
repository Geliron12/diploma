import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from source import logging
from source.preprocess.music21jsb import preprocess_music21
from source.preprocess.encode import encode_songs_data, get_density_bins

logger = logging.create_logger("datasetcreator")


class DatasetCreator:

    def __init__(self,
        dataset_name,
        encoding_method,
        json_data_method,
        window_size_bars,
        hop_length_bars,
        density_bins_number,
        transpositions_train,
        permute_tracks):

        self.dataset_name = dataset_name
        self.encoding_method = encoding_method
        self.json_data_method = json_data_method
        self.window_size_bars = window_size_bars
        self.hop_length_bars = hop_length_bars
        self.density_bins_number = density_bins_number
        self.transpositions_train = transpositions_train
        self.permute_tracks = permute_tracks

    def create(self, datasets_path, overwrite=False):

        # Проверка на существование каталога
        if not os.path.exists(datasets_path):
            os.mkdir(datasets_path)

        # Проверка существования датасета
        dataset_path = os.path.join(datasets_path, self.dataset_name)
        if os.path.exists(dataset_path) and overwrite is False:
            logger.info("Dataset already exists.")
            return
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        json_data_method = None
        if self.json_data_method == "preprocess_music21":
            json_data_method = preprocess_music21
        elif callable(self.json_data_method):
            json_data_method = self.json_data_method
        else:
            error_string = f"Unexpected {self.json_data_method}."
            logger.error(error_string)
            raise Exception(error_string)

        songs_data_train, songs_data_valid = json_data_method()

        density_bins = get_density_bins(
            songs_data_train,
            self.window_size_bars,
            self.hop_length_bars,
            self.
            density_bins_number
        )

        # Обработка и сохраниение тренировочных данных
        token_sequences_train = encode_songs_data(
            songs_data_train,
            transpositions=self.transpositions_train,
            permute=self.permute_tracks,
            window_size_bars=self.window_size_bars,
            hop_length_bars=self.hop_length_bars,
            density_bins=density_bins,
            bar_fill=self.encoding_method == "mmmbar"
        )
        dataset_path_train = os.path.join(dataset_path, "token_sequences_train.txt")
        self.__save_token_sequences(token_sequences_train, dataset_path_train)
        logger.info(f"Saved training data to {dataset_path_train}.")

        # Валидационных данных
        token_sequences_valid = encode_songs_data(
            songs_data_valid,
            transpositions=[0],
            permute=self.permute_tracks,
            window_size_bars=self.window_size_bars,
            hop_length_bars=self.hop_length_bars,
            density_bins=density_bins,
            bar_fill=self.encoding_method == "mmmbar"
        )
        dataset_path_valid = os.path.join(dataset_path, "token_sequences_valid.txt")
        self.__save_token_sequences(token_sequences_valid, dataset_path_valid)
        logger.info(f"Saved validation data to {dataset_path_valid}.")

        # Токенайзер
        tokenizer = self.__create_tokenizer([dataset_path_train, dataset_path_valid])
        tokenizer_path = os.path.join(dataset_path, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        logger.info(f"Saved tokenizer to {tokenizer_path}.")

    def __save_token_sequences(self, token_sequences, path):
        with open(path, "w") as file:
            for token_sequence in token_sequences:
                print(" ".join(token_sequence), file=file)

    def __create_tokenizer(self, files):

        # Создание, обучение, сохранение токенайзера
        print("Preparing tokenizer...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        tokenizer.train(files=files, trainer=trainer)
        return tokenizer

class JSBDatasetCreatorTrack(DatasetCreator):

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="jsb_mmmtrack",
            encoding_method="mmmtrack",
            json_data_method="preprocess_music21",
            window_size_bars=2,
            hop_length_bars=2,
            density_bins_number=5,
            transpositions_train=list(range(-12, 13)),
            permute_tracks=True,
            **kwargs
        )

class JSBDatasetCreatorBar(DatasetCreator):

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="jsb_mmmbar",
            encoding_method="mmmbar",
            json_data_method="preprocess_music21",
            window_size_bars=2,
            hop_length_bars=2,
            density_bins_number=5,
            transpositions_train=list(range(-12, 13)),
            permute_tracks=True,
            **kwargs
        )