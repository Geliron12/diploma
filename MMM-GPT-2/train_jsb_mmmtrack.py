import os
import sys
from source import datasetcreator
from source import mmmtrainer


# Создание датасета
dataset_creator = datasetcreator.JSBDatasetCreatorTrack()
dataset_creator.create(datasets_path=os.path.join("datasets"), overwrite=False)

# Обучение модели
trainer = mmmtrainer.JSBTrack(
    tokenizer_path = os.path.join("datasets", "jsb_mmmtrack", "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", "jsb_mmmtrack", "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", "jsb_mmmtrack", "token_sequences_valid.txt")],
    pad_length=768,
    shuffle_buffer_size=10000,
    batch_size=4,
    epochs=10,
)
trainer.train(
    output_path=os.path.join("training/jsb_mmmtrack"),
    simulate="simulate" in sys.argv
    )
