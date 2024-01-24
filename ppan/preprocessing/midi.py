import warnings
from pathlib import Path
from typing import Optional
from typing import Tuple, Union

import mido
import numpy as np
import numpy.typing as npt
import partitura as pt
import partitura.utils as ptu
from partitura.performance import Performance
from tqdm import tqdm


class PPAnMidi:
    """
    Class for handling midi operations. Things like tokenization and loading.
    """
    TIME_DIV = 120
    # Convert midi notes to their names, taken from here:
    # https://gist.github.com/devxpy/063968e0a2ef9b6db0bd6af8079dad2a
    NOTES = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#',
             'b']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)
    PIANO_SHIFT = 21

    def __init__(self):
        self.midi_filepath = None
        self._performance = None
        self._pianoroll = None
        self.midi_file = None
        self._note_array = None

        self.time_div = self.TIME_DIV
        self.notes = self.NOTES
        self.octaves = self.OCTAVES
        self.notes_in_octave = self.NOTES_IN_OCTAVE
        self.piano_shift = self.PIANO_SHIFT

    def number_to_note(self, number: int) -> str:
        octave = number // self.notes_in_octave
        note = self.notes[number % self.notes_in_octave]

        return note + str(octave)

    def note_to_number(self, note: str) -> int:
        octave = int(note[-1])
        note_no = self.NOTES.index(note[:-1])

        return octave * self.notes_in_octave + note_no + self.piano_shift

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
        self._note_array = None

    @property
    def note_array(self):
        if self._note_array is None:
            self._note_array = self.performance.note_array()
        return self._note_array

    @property
    def duration(self):
        note_offs = [
            i['note_off'] for i in self.performance.performedparts[0].notes
        ]
        return max(note_offs)

    @property
    def performance(self) -> Performance:
        if self._performance is None:
            self._performance = pt.load_performance_midi(self.midi_filepath)

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
                    time_div=self.time_div,
                    remove_silence=False,
                    binary=True
                ).toarray().astype(bool)
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
        start = int(timestamps[0] / (1/self.time_div))
        end = int(timestamps[1] / (1/self.time_div))

        return self.pianoroll[:, start:end]

    def midi_to_notes(self,
                      timestamps: Tuple[float, float]) -> npt.NDArray[int]:
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
        notes = np.array([
            i['midi_pitch'] for i in self.performance[0].notes if
            timestamps[1] > i['note_on'] > timestamps[0] or
            i['note_on'] < timestamps[0] < i['note_off']
        ])-self.piano_shift
        if any(notes > 87) or any(notes < 0):
            raise AttributeError("The note array seems to be invalid")
        return notes

    def midi_to_onset_offset_vec(self,
                      timestamps: Tuple[float, float]) -> npt.NDArray[int]:
        """
        Generate a pianoroll slice of all onsets and offsets played between two
        timestamps. The first half of the vector corresponds to onsets, while
        the second half corresponds to offsets.

        Parameters
        ----------
        timestamps : Tuple[float, float]
            start and end of the window in seconds

        Returns
        -------
        note_vec : npt.NDArray
        """
        res_array = np.zeros(shape=128*2, dtype=np.single)
        [
            res_array.put(1, i['midi_pitch']-self.piano_shift) for i in self.performance[0].notes if
            timestamps[1] > i['note_on'] > timestamps[0]
        ]
        [
            res_array.put(1, i['midi_pitch']-self.piano_shift+self.PIANO_SHIFT) for i in
            self.performance[0].notes if
            timestamps[1] > i['note_off'] > timestamps[0]
        ]
        return res_array

    def notes_to_sentence(self, notes):
        """
        Convert a note_vec to a list of notes as strings.

        Parameters
        ----------
        notes : npt.NDArray[int]

        Returns
        -------
        notes : List[str]
        """
        notes = [self.number_to_note(i) for i in notes]
        notes = " ".join(notes)

        return notes

    def sentence_to_note_vec(self, sentence: list[str]):
        note_array = np.zeros(shape=(128, 1), dtype=bool)
        if not sentence:
            return note_array
        elif not sentence[0]:
            return note_array

        split_sentence = sentence[0].split(" ")
        notes = [self.note_to_number(i) for i in split_sentence]
        for i in notes:
            note_array[i, :] = 1
        return note_array

    def sentences_to_pianoroll(self, sentences: list[list]):
        pianoroll = np.array([self.sentence_to_note_vec(i) for i in sentences],
                             dtype=bool)
        pianoroll = pianoroll.T
        return np.squeeze(pianoroll)



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
            sentence = loader.midi_to_notes((time, time+window_size))
            sentences.append((sentence, str(sample_no+start)))
    return sentences
