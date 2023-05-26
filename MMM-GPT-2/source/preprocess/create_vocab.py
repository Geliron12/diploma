from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import os
import json

tokenizer_path = os.path.join("datasets", "jsb_mmmtrack", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
# Разбиваем общий словарь на соответсвующие токены
vocab = tokenizer.vocab
insts = set()
dens = set()
note_ons = set()
delts = set()

for word in vocab.keys():
    if 'INST=' in word:
        insts.update([word])
    elif 'DENSITY=' in word:
        dens.update([word])
    elif 'NOTE_ON=' in word:
        note_ons.update([word])
    elif 'TIME_DELTA=' in word:
        delts.update([word])


insts_path = os.path.join("datasets", "jsb_mmmtrack", "insts.json")

with open(insts_path, "w") as outfile:
    json.dump(list(insts), outfile)

dens_path = os.path.join("datasets", "jsb_mmmtrack", "dens.json")

with open(dens_path, "w") as outfile:
    json.dump(list(dens), outfile)

note_ons_path = os.path.join("datasets", "jsb_mmmtrack", "note_ons.json")

with open(note_ons_path, "w") as outfile:
    json.dump(list(note_ons), outfile)

delts_path = os.path.join("datasets", "jsb_mmmtrack", "delts.json")

with open(delts_path, "w") as outfile:
    json.dump(list(delts), outfile)