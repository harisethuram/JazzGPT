import torch
import numpy as np
import sqlite3
import os.path
import pickle

KEYS = 12
CHORUS_COUNT = 6
# Global compression flag... very bad style, yes
COMPRESS = None

# harmony is the same
def get_harmony():
    """
    - returns filtered and augmented harmony (N, 2) i.e. with only songs in 4/4 and the uncommon chords
    replaced with 'unk'
    """
    con = sqlite3.connect("wjazzd.db")
    cur = con.cursor()

    # only select 4/4
    harmony_x = cur.execute(f"SELECT b.melid, b.chord from beats b JOIN solo_info s ON b.melid=s.melid "+ 
                            f"WHERE (s.signature='4/4' AND s.chorus_count<{CHORUS_COUNT})")
    harmony = np.array(harmony_x.fetchall())

    # fill in blank spaces
    prev_tok = "start"
    cur_melid = '1'
    for i in range(harmony.shape[0]):
        if harmony[i, 0] != cur_melid:
            cur_melid = harmony[i, 0]
            if harmony[i,1] == '':
                prev_tok = "start"
            else:
                prev_tok = harmony[i,1]
        
        if harmony[i,1] != '':
            prev_tok = harmony[i,1]
        else:
            harmony[i,1] = prev_tok
    
    harmony_b = make_batch(harmony, mode="har")

    # augment harmony
    harmony_aug = np.full((harmony_b.shape[0] * KEYS, harmony_b.shape[1], 1), '').astype(object)

    for i in range(harmony_b.shape[0]):
        for j in range(harmony_b.shape[1]):
            augs = None
            if harmony_b[i, j, 0] != '' and harmony_b[i, j, 0] != 'NC' and harmony_b[i, j, 0] != 'start':
                augs = augment_harmony(harmony_b[i, j, 0])
            elif harmony_b[i, j, 0] == 'start':
                augs = np.tile(np.array(['start']), KEYS)
            elif harmony_b[i, j, 0] == '':
                break
            elif harmony_b[i, j, 0] == 'NC':
                augs = np.tile(np.array(['unk']), KEYS)
            
            # get indices to modify
            indices = np.arange(0, stop=KEYS, step=1) * harmony_b.shape[0] + i
            harmony_aug[indices, j, 0] = augs
    harmony = harmony_aug

    # frequencies
    harmony_vocab, counts = np.unique(harmony[:, :, 0], return_counts=True)
    freqs = np.array((harmony_vocab, counts)).T
    freqs_dict = {row[0]: int(row[1]) for row in freqs}

    # unks
    for i in range(harmony.shape[0]):
        for j in range(harmony.shape[1]):
            if harmony[i, j, 0] == '':
                break
            if freqs_dict[harmony[i, j, 0]] < 20:
                harmony[i, j, 0] = 'unk'
    harmony_vocab, counts = np.unique(harmony[:, :, 0], return_counts=True)
    freqs = np.array((harmony_vocab, counts)).T
    freqs_dict = {row[0]: int(row[1]) for row in freqs}
    return harmony_vocab, harmony


def augment_harmony(chord):
    """
    - take in string representing chord
    - returns all 12 transpositions of chord
    """
    transpose = {
        'A': ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab'],
        'A#': ['Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A'],
        'Bb': ['Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A'],
        'B': ['B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb'],
        'Cb': ['B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb'],
        'C': ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'],
        'B#': ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'],
        'Db': ['Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C'],
        'C#': ['Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C'],
        'D': ['D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db'],
        'Eb': ['Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D'],
        'D#': ['Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D'],
        'E': ['E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb'],
        'Fb': ['E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb'],
        'F': ['F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E'],
        'E#': ['F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E'],
        'F#': ['Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F'],
        'Gb': ['Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F'],
        'G': ['G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb'],
        'Ab': ['Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G'],
        'G#': ['Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G'],
    }
    # in case of slash chords
    chord = chord.split("/")
    upper = chord[0]
    bass = ''
    if len(chord) > 1:
        bass = chord[1]
    keys = transpose.keys()
    key=None
    fluff = '' # extensions and chord quality
    if upper[:2] in keys:
        key = upper[:2]
        fluff = upper[2:]
    elif upper[:1] in keys:
        key = upper[:1]
        fluff = upper[1:]
    key_b = ''
    fluff_b = ''
    if bass != '':
        if bass[:2] in keys:
            key_b = bass[:2]
            fluff_b = bass[2:]
        elif bass[:1] in keys:
            key_b = bass[:1]
            fluff_b = bass[1:]

    transpositions = []

    for i in range(KEYS):
        bass_transpose = ''
        if bass != '':
            bass_transpose = "/" + transpose[key_b][i] + fluff_b
        transpositions.append(transpose[key][i]+fluff+bass_transpose)
    return np.array(transpositions)

def get_melody():
    """
    - returns filtered harmony (N, 5) i.e. with only songs in 4/4 and the uncommon notes replaced with (0, 0, 0, 0)
    """
    con = sqlite3.connect("wjazzd.db")
    cur = con.cursor()

    melody_x = cur.execute(f"SELECT m.melid, m.pitch, m.onset, m.duration, m.beat, m.tatum, m.division, m.beatdur, s.avgtempo from melody m JOIN solo_info s ON m.melid=s.melid WHERE (s.signature='4/4' AND s.chorus_count<{CHORUS_COUNT})")
    melody = np.array(melody_x.fetchall()) # N,5, melid pitch onset duration beat tatum division beatdur
    PRECISION = 24 # to what-th note will I be rounding
    num_keys = 12

    # batch songs
    melody_b = make_batch(melody, mode="mel")
    
    # quantize songs
    melody_q = np.zeros((melody_b.shape[0], melody_b.shape[1]*2, 2)) # pitch duration (pitch = 0 is rest)
    beatdurs = melody_b[:, :, -2]
    avgtempo = melody_b[:, :, -1]
    first_beat = melody_b[:, 0, 3]
    first_tatum = melody_b[:, 0, 4]
    first_division = melody_b[:, 0, 5]
    pitches = melody_b[:, :, 0]
    first_start_offset = (first_beat-1) + (first_tatum - 1) / first_division
    durations = np.nan_to_num(melody_b[:, :, 2] / (beatdurs), 0) # duration of every note
    durations_q = np.round(durations * PRECISION) / PRECISION

    onsets = np.nan_to_num(np.maximum(melody_b[:, :, 1] - np.expand_dims(melody_b[:, 0, 1], axis=1), 0) / (60/avgtempo), 0)
    onsets_q = np.round(onsets * PRECISION) / PRECISION + np.expand_dims(first_start_offset, axis=1)
    sum_thing = np.nan_to_num(onsets_q + durations_q, 0) # at what beat each note ends
    rests = np.round(np.maximum(onsets_q - np.roll(sum_thing, shift=1, axis=1), 0) * PRECISION) / PRECISION     

    rests = np.nan_to_num(rests, 0)
    durations_q = np.nan_to_num(durations_q, 0)
    
    melody_q[:, ::2, 1] = rests
    melody_q[:, 1::2, 1] = durations_q
    melody_q[:, 1::2, 0] = pitches

    non_zeros = np.zeros_like(melody_q)
    thing = np.copy(melody_q[:, :, 1])
    inds = np.where(thing != 0)
    curs = np.zeros((melody_q.shape[0])).astype(int)

    for i in range(len(inds[0])):
        non_zeros[inds[0][i], curs[inds[0][i]], :] = melody_q[inds[0][i], inds[1][i], :]
        curs[inds[0][i]] += 1
    
    # dont need none of that
    melody = non_zeros

    # augment melody
    melody_aug = np.zeros((melody.shape[0]*num_keys, melody.shape[1], 2))
    for i in range(melody.shape[0]):
        for j in range(melody.shape[1]):
            if melody[i, j, 0] == 0 and melody[i, j, 1] == 0:
                break
            else:
                augs = augment_melody(melody[i, j, :])
                indices = np.arange(0, stop=num_keys, step=1) * melody.shape[0] + i
                melody_aug[indices, j, :] = augs
            
    melody = melody_aug
    flattened_melody = np.reshape(melody, (melody.shape[0]*melody.shape[1], melody.shape[2]))
    melody_vocab, counts = np.unique(flattened_melody, axis=0, return_counts=True)
    freqs = np.concatenate((melody_vocab, np.expand_dims(counts, axis=1)), axis=1)
    return melody_vocab, melody

def augment_melody(note):
    """
    - takes in note (2, ) np array where first index is pitch
    - returns all 12 transpositions upwards an octave of the note (12, 4)
    """
    note = np.expand_dims(note, axis=1)
    transpositions = np.tile(note, 12, )
    if note[0] != 0:
        amount = np.arange(start=0, stop=12, step=1)
    else:
        amount = np.zeros((12))
    transpositions[0, :] += amount
    return transpositions.T

def make_batch(songs, mode):
    """
        - takes in songs in dataset in form (T, d+1) where T is total number of events, and d is representation dimension (+1 as zeroeth column is the song id)
        - returns them batched by songs (N, M, d) where N is num songs, M is max length of a song
    """
    id = songs[:, 0].astype(int)
    song_ids = np.unique(id)
    num_songs = song_ids.shape[0]

    max_length = np.max(np.bincount(songs[:, 0].astype(int)))
    lengths = np.bincount(songs[:, 0].astype(int))

    dim = songs.shape[1] - 1

    batched = None
    if mode == "mel":
        batched = np.zeros((num_songs, max_length, dim))
    elif mode == "har":
        batched = np.full((num_songs, max_length, dim), "", dtype=object)

    for i in range(num_songs):
        song = np.squeeze(songs[np.where(id==song_ids[i]), 1:], axis=0)        
        batched[i, :song.shape[0]] = song
        
    return batched

def indices_har(songs, vocab):
    """
    - takes in batched harmony of songs in shape and vocab
    - returns their indices with dictionary
    """

    # Adds the end token
    song_to_i = {s : i for i, s in enumerate(vocab)}
    song_to_i['end'] = len(vocab)

    i_to_song = {i : s for i, s in enumerate(vocab) }
    i_to_song[len(vocab)] = 'end'
    
    inds = np.zeros((songs.shape[0], songs.shape[1]))
    for i in range(songs.shape[0]):
        in_song = True
        for j in range(songs.shape[1]):
            s = songs[i, j, 0]
            inds[i, j] = song_to_i[s]
            if inds[i, j] == 0 and in_song:
                in_song = False
                inds[i, j] = len(vocab)

    global COMPRESS
    if not COMPRESS:
        return inds, song_to_i, i_to_song
    
    # Removes duplicates
    jagged = []
    for i in range(inds.shape[0]):
        songinds = inds[i]
        jagged.append(
            songinds[np.concatenate( ([True], np.diff(songinds) != 0) )]
        )

    # Pads everything through
    lengths = [len(arr) for arr in jagged]
    m = max(lengths)
    padding = [m - l for l in lengths]
    padded = [np.concatenate((arr, [0] * p)) for p, arr in zip(padding, jagged)]
    inds = np.stack(padded)
        
    return inds, song_to_i, i_to_song

def indices_mel(songs, vocab):
    """
    - takes in batched melody of songs in shape and vocab
    - returns their indices with dictionary
    """
    vocab = np.concatenate( (vocab, [(200., 0.)]) )
        
    song_to_i = { tuple(s) : i for i, s in enumerate(vocab) }
    endtk = (max(vocab[:, 0]), 0.0)
    song_to_i[endtk] = len(vocab)
    
    i_to_song = { i:s for i, s in enumerate(vocab) }
    i_to_song[len(vocab)] = endtk
    
    startk = song_to_i[(200, 0)]

    inds = np.zeros((songs.shape[0], songs.shape[1] + 1))
    for i in range(songs.shape[0]):
        in_song = True
        inds[i, 0] = startk
        for j in range(0, songs.shape[1]):
            s = tuple(songs[i, j, :])
            inds[i, j + 1] = song_to_i[tuple(s)]
            if inds[i, j + 1] == song_to_i[ (0, 0) ] and in_song:
                in_song = False
                inds[i, j + 1] = len(vocab)
    
    return inds, song_to_i, i_to_song

def main(create=False, compress=False):
    '''
    Loads the data. If create is False, will first check for existing
    data before creating anew. If compress is False, will not compress
    multiple instances of the same chord into one token.
    '''
    global COMPRESS
    COMPRESS = compress
    path = ''
    if COMPRESS:
        print('In compressed mode')
        PATH = './compressed/'
    else:
        print('In uncompressed mode')
        PATH = './uncompressed/'
        
    if ((not os.path.isfile(f'{PATH}harmony.pt')) and (not os.path.isfile(f'{PATH}melody.pt'))) or create==True:
        print("Creating data")
        harmony_vocab, harmony = get_harmony()
        melody_vocab, melody = get_melody()
        
        ih, har_to_i, i_to_har = indices_har(harmony, harmony_vocab)
        im, mel_to_i, i_to_mel = indices_mel(melody, melody_vocab)
        # save values as files
        print("Pickling data")
        with open(f'{PATH}melody.pt', 'wb') as f:
            torch.save(im, f)
        with open(f'{PATH}harmony.pt', 'wb') as f:
            torch.save(ih, f)
        with open(f'{PATH}har_to_i.pkl', 'wb') as f:
            pickle.dump(har_to_i, f)
        with open(f'{PATH}i_to_har.pkl', 'wb') as f:
            pickle.dump(i_to_har, f)
        with open(f'{PATH}mel_to_i.pkl', 'wb') as f:
            pickle.dump(mel_to_i, f)
        with open(f'{PATH}i_to_mel.pkl', 'wb') as f:
            pickle.dump(i_to_mel, f)
    else:
        print("loading data")
        ih = torch.load(f'{PATH}harmony.pt')
        im = torch.load(f'{PATH}melody.pt')
        
        with open(f'{PATH}har_to_i.pkl', 'rb') as fp:
            har_to_i = pickle.load(fp)
        with open(f'{PATH}i_to_har.pkl', 'rb') as fp1:
            i_to_har = pickle.load(fp1)
        with open(f'{PATH}mel_to_i.pkl', 'rb') as fp2:
            mel_to_i = pickle.load(fp2)
        with open(f'{PATH}i_to_mel.pkl', 'rb') as fp3:
            i_to_mel = pickle.load(fp3)

    return ih, har_to_i, i_to_har, im, mel_to_i, i_to_mel

if __name__ == '__main__':
    main(True)
else: print('Importing altrepalt.py')