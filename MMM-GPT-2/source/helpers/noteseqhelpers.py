import note_seq
## Длина ноты и длина такта
NOTE_LENGTH_16TH_120BPM = 0.25 * 60 / 120
BAR_LENGTH_120BPM = 4.0 * 60 / 120

#функция, подстраивающая темп под таргет темп
def set_note_sequence_tempo(note_sequence, target_tempo):

    raise_exception_on_multiple_tempos(note_sequence)

    current_tempo = note_sequence.tempos[0].qpm
    multiplier = current_tempo / target_tempo

    note_sequence.tempos[0].qpm = target_tempo

    note_sequence.total_time *= multiplier

    for note in note_sequence.notes:
        note.start_time *= multiplier
        note.end_time *= multiplier

    return note_sequence

# функция, разделяющая последовательности нот на такты
def split_note_sequence_into_bars(note_sequence, absolute_times, threshold=0.0, quantized=False):

    raise_exception_on_multiple_tempos(note_sequence)

    qpm = note_sequence.tempos[0].qpm

    bar_length = 4 * 60.0 / qpm

    if not quantized:
        bars = note_sequence_to_bars(note_sequence, threshold=threshold)
    else:
        bars = note_sequence_to_bars_quantized(note_sequence)
    assert len(bars) != 0

    note_sequences = bars_to_note_sequences(bars, qpm, bar_length, absolute_times)
    assert len(note_sequences) != 0

    return note_sequences


def note_sequence_to_bars(note_sequence, threshold):

    qpm = note_sequence.tempos[0].qpm

    bar_length = 4 * 60.0 / qpm

    bars_number = int(round(note_sequence.total_time / bar_length))

    bars = []
    start_time = 0.0
    processed_notes = []
    for index in range(bars_number):

        if len(processed_notes) >= len(note_sequence.notes):
            break

        end_time = start_time + bar_length

        notes = []
        for note in note_sequence.notes:
            if note.start_time >= start_time - threshold and note.start_time < end_time - threshold:
                notes += [note]

        bars += [notes]

        start_time = end_time


    return bars


def note_sequence_to_bars_quantized(note_sequence, steps_per_bar=16):

    notes_to_process = sorted(note_sequence.notes, key=lambda note: note.quantized_start_step)

    bars = []
    step_start = 0
    steps_maximum = steps_per_bar * 100
    for step_start, step_end in zip(range(0, steps_maximum, steps_per_bar), range(steps_per_bar, steps_maximum, steps_per_bar)):

        if len(notes_to_process) == 0:
            break

        notes_found = []
        for note in notes_to_process:
            if note.quantized_start_step >= step_start and note.quantized_start_step < step_end:
                notes_found += [note]

            elif note.quantized_start_step >= step_end:
                break

        for note in notes_found:
            notes_to_process.remove(note)

        bars += [notes_found]

    return bars

# нотные последовательности из тактов
def bars_to_note_sequences(bars, qpm, bar_length, absolute_times):

    note_sequences = []

    for bar_index, bar in enumerate(bars):
        note_sequence_bar = empty_note_sequence(qpm=qpm, total_time=bar_length)

        for note in bar:
            new_note = note_sequence_bar.notes.add()
            new_note.CopyFrom(note)

            if not absolute_times:
                new_note.start_time -= bar_index * bar_length
                new_note.end_time -= bar_index * bar_length
                new_note.quantized_start_step -= bar_index * 16
                new_note.quantized_end_step -= bar_index * 16

        note_sequences.append(note_sequence_bar)

    return note_sequences


def clip_quantized_steps(note_sequence, steps):
    for note in note_sequence.notes:
        if note.quantized_start_step < 0:
            note.quantized_start_step = 0
        if note.quantized_start_step >= steps:
            note.quantized_start_step = steps - 1
        if note.quantized_end_step < 1:
            note.quantized_end_step = 1
        if note.quantized_end_step > steps:
            note.quantized_end_step = steps
    return note_sequence

# создание пустой музыкальной последовательности
def empty_note_sequence(qpm=120.0, total_time=0.0):
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    note_sequence.total_time = total_time
    return note_sequence

## exception в случае если несколько показателей темпа
def raise_exception_on_multiple_tempos(note_sequence):
    if len(note_sequence.tempos) != 1:
        error_message = f"Too many tempos: {len(note_sequence.tempos)}"
        raise Exception(error_message)
