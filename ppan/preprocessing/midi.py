from typing import Tuple, Union
from pathlib import Path
import warnings

import mido
from mido.midifiles.meta import KeySignatureError
import numpy as np
import numpy.typing as npt
import partitura as pt
import partitura.utils as ptu
from partitura.performance import Performance
from typing import Optional
from tqdm import tqdm


class PPAnMidi:
    """
    Class for handling midi operations. Things like tokenization and loading.
    """
    TIME_DIV = 120
    # Convert midi notes to their names, taken frome here:
    # https://gist.github.com/devxpy/063968e0a2ef9b6db0bd6af8079dad2a
    NOTES = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#',
             'b']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)

    def __init__(self, max_len: Optional[int] = None):
        """
        Parameters
        ----------
        max_len : Optional[int]
            if max_len is not provided the tokenizer will not be initialized
        """
        self.midi_filepath = None
        self._performance = None
        self._pianoroll = None
        self.vocab_len = None

        if max_len is not None:
            self.midi_file = None

    def number_to_note(self, number: int) -> str:
        octave = number // self.NOTES_IN_OCTAVE
        note = self.NOTES[number % self.NOTES_IN_OCTAVE]

        return note + str(octave)

    def set_midi(self, midi_filepath):
        """
        Set a new midi file in the object. Resets all the cached objects such
        as pianoroll and performance.

        Parameters
        ----------
        midi_filepath : str
        """
        self.midi_filepath = midi_filepath
        self.midi_file = mido.MidiFile(midi_filepath)
        self._performance = None
        self._pianoroll = None

    @property
    def duration(self):
        note_offs = [i['note_off'] for i in self.performance.performedparts[0].notes]
        return max(note_offs)

    @property
    def performance(self) -> Performance:
        if self._performance is None:
            self._performance = pt.load_performance_midi(self.midi_filepath)
            # Remove peddle, as we're only interested in whether the key is
            # pressed
            self._performance.performedparts[0].controls = []
            for i in self._performance.performedparts[0].notes:
                i['sound_off'] = i['note_off']
        return self._performance

    @property
    def pianoroll(self) -> npt.NDArray:
        if self._pianoroll is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pianoroll = ptu.compute_pianoroll(
                    self.performance,
                    piano_range=True,
                    time_unit="sec",
                    time_div=self.TIME_DIV,
                    remove_silence=False,
                    binary=True
                ).toarray()
        return self._pianoroll

    def pianoroll_window(self, timestamps: Tuple[float, float]):
        """
        Extract a window from the midi pianoroll given start and end times and
        return it as an array.

        Parameters
        ----------
        timestamps : Tuple[float, float]

        Returns
        -------
        pianoroll_window : npt.NDArray
        """
        start = int(timestamps[0] // (1/self.TIME_DIV))
        end = int(timestamps[1] // (1/self.TIME_DIV))

        return self.pianoroll[:, start:end]

    def midi_to_vec(self, timestamps: Tuple[float, float]):
        """
        Generate a vector slice of all notes played between two timestamps.

        Parameters
        ----------
        timestamps : Tuple[float, float]
            start and end of the window in seconds

        Returns
        -------
        note_vec : npt.NDArray
        """
        pianoroll_seg = self.pianoroll_window(timestamps=timestamps)
        nonzero = np.argwhere(pianoroll_seg != 0)
        note_vec = np.zeros(shape=(pianoroll_seg.shape[0], 1), dtype=bool)

        if nonzero.shape == (0, 2):
            return note_vec

        for i in nonzero[:, 0]:
            note_vec[i, :] = 1

        return note_vec

    def note_vec_to_sentence(self, note_vec):
        """
        Convert a note_vec to a list of notes as strings.

        Parameters
        ----------
        note_vec : npt.NDArray[bool]

        Returns
        -------
        notes : List[str]
        """
        notes = np.argwhere(np.squeeze(note_vec)).tolist()
        notes = [self.number_to_note(i[0]) for i in notes]
        notes = " ".join(notes)
        return notes


def extract_sentences(samples: list[Union[str, Path]],
                      window_size: Optional[float] = None,
                      stride: Optional[float] = None,
                      start: int = None):
    """
    Go through a list of midi files and extract "sentences". Each sentence is
    a list of notes within a generated window.
    The next sentence starts at the next stride.

    Parameters
    ----------
    samples : list[str]
    window_size : Optional[float]
        Size of the window in seconds. Defaults to 0.02
    stride : Optional[float]
        Distance between sentences in seconds. Defaults to 0.3.
    start : int
        Number to start at as file index.

    Returns
    -------
    sentences : list[str]
    """
    if window_size is None:
        window_size = 0.2
    if stride is None:
        stride = 0.9
    if start is None:
        start = 0

    loader = PPAnMidi()
    sentences = []
    for sample_no, sample in enumerate(tqdm(samples,
                                            desc="Extracting sentences "
                                                 "from MIDI")):
        try:
            loader.set_midi(sample)
            midi_duration = loader.duration
        except:
            continue

        times = np.arange(0, midi_duration, stride)
        for time in times:
            note_vec = loader.midi_to_vec((time, time+window_size))
            sentence = loader.note_vec_to_sentence(note_vec=note_vec)
            sentences.append((sentence, str(sample_no+start)))
    return sentences
