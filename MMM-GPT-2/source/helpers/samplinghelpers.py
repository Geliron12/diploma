import note_seq
import random
import json
import os
from source.helpers.noteseqhelpers import (
    empty_note_sequence,
    NOTE_LENGTH_16TH_120BPM,
    BAR_LENGTH_120BPM
)

# функция для обработки последовательности токенов и перевода ее в последовательность нот
def render_token_sequence(token_sequence, midi_path, use_program=True, use_drums=True):
    note_sequence = token_sequence_to_note_sequence(token_sequence, use_program=use_program, use_drums=use_drums)
    note_seq.plot_sequence(note_sequence)
    note_seq.midi_io.note_sequence_to_midi_file(note_sequence,midi_path)

# вывод последовательности токенов
def print_token_sequence(token_sequence, priming_samples_number=None):

    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()
    assert isinstance(token_sequence, list)

    indent_level = 0
    result = ""
    for token_index, token in enumerate(token_sequence):

        if priming_samples_number is not None:
            if token_index < priming_samples_number:
                first_character = "P "
            else:
                first_character = "  "
        else:
            first_character = ""

        if token in ["PIECE_END", "TRACK_END", "BAR_END"]:
            indent_level -= 1

        result += first_character + f"{token_index:04d} " + "  " * indent_level + token + "\n"

        if token in ["PIECE_START", "TRACK_START", "BAR_START"]:
            indent_level += 1

    print(result)


def get_priming_token_sequence(data_path, stop_on_track_end=None, stop_after_n_tokens=None, return_original=False):

    # Случайная последовательность токенов из файла
    lines = open(data_path, "r").readlines()
    token_sequence = random.choice(lines)

    result_tokens = []
    track_end_index = 0
    for token_index, token in enumerate(token_sequence.split()):
        result_tokens += [token]

        if stop_on_track_end == track_end_index and token == "TRACK_END":
            break

        if token == "TRACK_END":
            track_end_index += 1

        if stop_after_n_tokens != 0 and token_index + 1 == stop_after_n_tokens:
            break

    result = " ".join(result_tokens)
    if not return_original:
        return result
    else:
        return result, token_sequence


def generate(model, tokenizer, token_sequence):

    input_ids = tokenizer.encode(token_sequence, return_tensors="pt")
    generated_sequence = model.generate(
        input_ids,
        max_length=1000,
        temperature=0.9,
    )
    generated_sequence = tokenizer.decode(generated_sequence[0])
    return generated_sequence

def generate_from_scratch(model, tokenizer, return_info=False):

    insts_path = os.path.join("datasets", "jsb_mmmtrack", "insts.json")
    with open(insts_path, "r") as outfile:
        insts = json.load(outfile)

    dens_path = os.path.join("datasets", "jsb_mmmtrack", "dens.json")

    with open(dens_path, "r") as outfile:
        dens = json.load(outfile)

    note_ons_path = os.path.join("datasets", "jsb_mmmtrack", "note_ons.json")

    with open(note_ons_path, "r") as outfile:
        note_ons = json.load(outfile)

    delts_path = os.path.join("datasets", "jsb_mmmtrack", "delts.json")

    with open(delts_path, "r") as outfile:
        delts = json.load(outfile)

    inst = random.choice(insts)
    den = random.choice(dens)
    note_on = random.choice(note_ons)
    delt = random.choice(delts)
    
    priming_sample = ('PIECE_START TRACK_START ' + inst + ' ' + den +
                      ' BAR_START '+ note_on + ' ' + delt)
    
    if return_info:

         return (generate(model, tokenizer, priming_sample), 
                 inst, den)

    return (generate(model, tokenizer, priming_sample))

def format_seq(note_seq):
    note_seq = note_seq.replace('PIECE_START ','') 
    note_seq = note_seq.replace(' [PAD]','') 
    note_seq = note_seq.replace('TRACK_END ','')
    note_seq_splitted = note_seq.split('TRACK_START ')
    note_seq_splitted.pop(0)
    for i in range(len(note_seq_splitted)):
        note_seq_splitted[i] = note_seq_splitted[i].split('BAR_END ')
        note_seq_splitted[i].pop(-1)
    return note_seq_splitted

def generate_long_sample(model, tokenizer, num_of_steps = 6):
    
    note_seq, inst, den = generate_from_scratch(model,tokenizer, return_info=True)

    note_seq = format_seq(note_seq)

    if len(note_seq)>4:
        return generate_long_sample(model, tokenizer, num_of_steps = 6)
    for i in range(num_of_steps):
        start_seq = ('PIECE_START ' + 'TRACK_START ' + inst +
                      ' ' + den + ' ' + note_seq[0][-1] + 'BAR_END ')
        generated_sample = generate(model, tokenizer, start_seq)
        generated_sample = format_seq(generated_sample)
        for j in range(len(note_seq)):
            note_seq[j].append(generated_sample[j][-1])

    for i in range(len(note_seq)):
        note_seq[i] = 'TRACK_START ' + 'BAR_END '.join(note_seq[i]) + 'BAR_END '

    final_seq = 'TRACK_END '.join(note_seq) + 'TRACK_END '
    return final_seq

def token_sequence_to_note_sequence(token_sequence, use_program=True, use_drums=True):

    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()

    note_sequence = empty_note_sequence()
    current_program = 1
    current_is_drum = False
    for token_index, token in enumerate(token_sequence):

        if token == "PIECE_START":
            pass
        elif token == "PIECE_END":
            print("The end.")
            break
        elif token == "TRACK_START":
            current_bar_index = 0
            pass
        elif token == "TRACK_END":
            pass
        elif token.startswith("INST"):
            current_instrument = token.split("=")[-1]
            if current_instrument != "DRUMS" and use_program:
                current_instrument = int(current_instrument)
                current_program = int(current_instrument)
                current_is_drum = False
            if current_instrument == "DRUMS" and use_drums:
                current_instrument = 0
                current_program = 0
                current_is_drum = True
        elif token == "BAR_START":
            current_time = current_bar_index * BAR_LENGTH_120BPM
            current_notes = {}
        elif token == "BAR_END":
            current_bar_index += 1
            pass
        elif token.startswith("NOTE_ON"):
            pitch = int(token.split("=")[-1])
            note = note_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + 4 * NOTE_LENGTH_16TH_120BPM
            note.pitch = pitch
            note.instrument = int(current_instrument)
            note.program = current_program
            note.velocity = 80
            note.is_drum = current_is_drum
            current_notes[pitch] = note
        elif token.startswith("NOTE_OFF"):
            pitch = int(token.split("=")[-1])
            if pitch in current_notes:
                note = current_notes[pitch]
                note.end_time = current_time
        elif token.startswith("TIME_DELTA"):
            delta = float(token.split("=")[-1]) * NOTE_LENGTH_16TH_120BPM
            current_time += delta
        elif token.startswith("DENSITY="):
            pass
        elif token == "[PAD]":
            pass
        else:
            assert False, token

    return note_sequence
