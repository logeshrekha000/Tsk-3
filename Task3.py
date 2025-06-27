from music21 import converter, instrument, note, chord, stream
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.utils import np_utils
import glob
import pickle
import random

# Step 1: Extract notes from MIDI files
def extract_notes_from_midi(folder="data"):
    notes = []
    for file in glob.glob(f"{folder}/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flat.notes
        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Step 2: Create LSTM model
def create_model(input_shape, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 3: Generate MIDI from predicted notes
def create_midi(prediction_output, filename="output.mid"):
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = len(output_notes)
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = len(output_notes)
            output_notes.append(new_note)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

# Main execution
if __name__ == "__main__":
    notes = extract_notes_from_midi("data")
    sequence_length = 100

    # Create mappings
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for note, number in note_to_int.items())

    # Prepare sequences
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    n_vocab = len(pitchnames)

    X = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    y = np_utils.to_categorical(network_output)

    model = create_model((X.shape[1], X.shape[2]), n_vocab)
    model.fit(X, y, epochs=5, batch_size=64)

    # Generate music
    start = random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    generated_notes = []

    for i in range(200):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        generated_notes.append(result)
        pattern.append(index)
        pattern = pattern[1:]

    create_midi(generated_notes, "output.mid")
