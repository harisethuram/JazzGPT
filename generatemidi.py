from midiutil import MIDIFile
import torch

def generate_midi(notes, file_name):
    """
    - takes in notes (d,2) tensor where each row is a note, col 0 is pitch, col 1 is duration
    - generates a midi file for the notes
    """
    # print(notes.shape)
    track = 0
    channel = 0
    time = 0
    tempo = 220
    volume = 100
    midi = MIDIFile(1)
    midi.addTempo(track, time, tempo)
    for i in range(1, notes.shape[0]):
        if notes[i, 1] == 0 and notes[i, 0] == 0: # i.e. reached padding
            break
        pitch = notes[i, 0]
        duration = notes[i, 1]
        if pitch != 0:
            midi.addNote(track, channel, int(notes[i, 0]), time, notes[i, 1], volume)
        time += duration
    string = file_name + ".mid"
    with open(string, "wb") as output_file:
        midi.writeFile(output_file)
