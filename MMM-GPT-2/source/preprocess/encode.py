import itertools
import numpy as np
import random
import json


def encode_songs_data(songs_data, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill):

    token_sequences = []

    for song_data in songs_data:
        token_sequences += encode_song_data(song_data, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill)

    return token_sequences


def encode_song_data(song_data, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill):

    token_sequences = []

    bars = get_bars_number(song_data)

    bar_indices = get_bar_indices(bars, window_size_bars, hop_length_bars)

    count = 0
    for (bar_start_index, bar_end_index), transposition in itertools.product(bar_indices, transpositions):

        token_sequence = []

        if bar_fill:
            track_data = random.choice(song_data["tracks"])
            bar_data = random.choice(track_data["bars"][bar_start_index:bar_end_index])
            bar_data_fill = {"events": bar_data["events"]}
            bar_data["events"] = "bar_fill"

        token_sequence += ["PIECE_START"]

        track_data_indices = list(range(len(song_data["tracks"])))
        if permute:
            random.shuffle(track_data_indices)

        for track_data_index in track_data_indices:
            track_data = song_data["tracks"][track_data_index]

            encoded_track_data = encode_track_data(track_data, density_bins, bar_start_index, bar_end_index, transposition)
            token_sequence += encoded_track_data

        if bar_fill:
            token_sequence += encode_bar_data(bar_data_fill, transposition, bar_fill=True)

        token_sequences += [token_sequence]
        count += 1

    return token_sequences


def encode_track_data(track_data, density_bins, bar_start_index, bar_end_index, transposition):

    tokens = []

    tokens += ["TRACK_START"]

    number = track_data["number"]

    if not track_data.get("drums", False):
        tokens += [f"INST={number}"]

    else:
        tokens += ["INST=DRUMS"]
        transposition = 0

    note_on_events = 0
    for bar_data in track_data["bars"][bar_start_index:bar_end_index]:
        if bar_data["events"] == "bar_fill":
            continue
        for event_data in bar_data["events"]:
            if event_data["type"] == "NOTE_ON":
                note_on_events += 1

    density = np.digitize(note_on_events, density_bins)
    tokens += [f"DENSITY={density}"]

    for bar_data in track_data["bars"][bar_start_index:bar_end_index]:
        tokens += encode_bar_data(bar_data, transposition)

    tokens += ["TRACK_END"]

    return tokens


def encode_bar_data(bar_data, transposition, bar_fill=False):
    tokens = []

    if not bar_fill:
        tokens += ["BAR_START"]
    else:
        tokens += ["FILL_START"]

    if bar_data["events"] == "bar_fill":
        tokens += ["FILL_IN"]
    else:
        for event_data in bar_data["events"]:
            tokens += [encode_event_data(event_data, transposition)]

    if not bar_fill:
        tokens += ["BAR_END"]
    else:
        tokens += ["FILL_END"]

    return tokens


def encode_event_data(event_data, transposition):
    if event_data["type"] == "NOTE_ON":
        return event_data["type"] + "=" + str(event_data["pitch"] + transposition)
    elif event_data["type"] == "NOTE_OFF":
        return event_data["type"] + "=" + str(event_data["pitch"] + transposition)
    elif event_data["type"] == "TIME_DELTA":
        return event_data["type"] + "=" + str(event_data["delta"])


def get_density_bins(songs_data, window_size_bars, hop_length_bars, bins):

    distribution = []
    for song_data in songs_data:
        bars = get_bars_number(song_data)

        bar_indices = get_bar_indices(bars, window_size_bars, hop_length_bars)
        for track_data in song_data["tracks"]:
            for bar_start_index, bar_end_index in bar_indices:

                count = 0
                for bar in track_data["bars"][bar_start_index:bar_end_index]:
                    count += len([event for event in bar["events"] if event["type"] == "NOTE_ON"])

                if count != 0:
                    distribution += [count]

    quantiles = []
    for i in range(100 // bins, 100, 100 // bins):
        quantile = np.percentile(distribution, i)
        quantiles += [quantile]
    return quantiles


def get_density_bins_from_json_files(json_paths, window_size_bars, hop_length_bars, bins):

    distribution = []
    for json_path in json_paths:

        song_data = json.load(open(json_path, "r"))

        bars = get_bars_number(song_data)

        bar_indices = get_bar_indices(bars, window_size_bars, hop_length_bars)
        for track_data in song_data["tracks"]:
            for bar_start_index, bar_end_index in bar_indices:

                count = 0
                for bar in track_data["bars"][bar_start_index:bar_end_index]:
                    count += len([event for event in bar["events"] if event["type"] == "NOTE_ON"])

                if count != 0:
                    distribution += [count]

    quantiles = []
    for i in range(100 // bins, 100, 100 // bins):
        quantile = np.percentile(distribution, i)
        quantiles += [quantile]
    return quantiles


def get_bars_number(song_data):
    bars = [len(track_data["bars"]) for track_data in song_data["tracks"]]
    bars = max(bars)
    return bars


def get_bar_indices(bars, window_size_bars, hop_length_bars):
    return list(zip(range(0, bars, hop_length_bars), range(window_size_bars, bars, hop_length_bars)))
