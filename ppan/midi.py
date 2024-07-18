import warnings
from random import choice
from typing import Tuple

import mido
import numpy as np
import numpy.typing as npt
import partitura as pt
import partitura.utils as ptu
import torch
from partitura.performance import Performance

from ppan.config import num_labels


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

    def __init__(self, n_frames: int, temporal_res: float,
                 lenience: int, percentage_negative: float = 0.01):

        self.midi_filepath = None
        self._performance = None
        self._pianoroll = None
        self.midi_file = None
        self._note_array = None
        self.percentage_negative = percentage_negative
        self.time_div = self.TIME_DIV
        self.notes = self.NOTES
        self.octaves = self.OCTAVES
        self.notes_in_octave = self.NOTES_IN_OCTAVE
        self.piano_shift = self.PIANO_SHIFT

        self._oo_array = None
        self.temporal_res = temporal_res
        self.lenience = lenience
        self.n_frames = n_frames

    @property
    def oo_array(self) -> npt.NDArray[np.bool_]:
        if self._oo_array is None:
            _oo_array = np.zeros(shape=(num_labels,
                                        self.n_frames),
                                 dtype=np.bool_)
            for note in self.performance.performedparts[0].notes:
                note_on_frame = round(note['note_on'] / self.temporal_res)
                _oo_array[
                note['midi_pitch'] - self.PIANO_SHIFT,
                note_on_frame - self.lenience:note_on_frame + self.lenience + 1
                ] = 1
            self._oo_array = _oo_array
        return self._oo_array

    def __call__(self, time: float | int, device: torch.device,
                 dtype: torch.dtype):
        if isinstance(time, float):
            return torch.tensor(
                self.oo_array[:, int(time // self.temporal_res)],
                device=device, dtype=dtype)
        else:
            return torch.tensor(self.oo_array[:, time],
                                device=device, dtype=dtype)

    def number_to_note(self, number: int) -> str:
        octave = number // self.notes_in_octave
        note = self.notes[number % self.notes_in_octave]

        return note + str(octave)

    def get_all_onsets(self):
        return [
            i['note_on'] for i in self.performance.performedparts[0].notes
        ]

    def get_all_offsets(self):
        return [
            i['note_off'] for i in self.performance.performedparts[0].notes
        ]

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
        self._oo_array = None
        return self

    @property
    def note_array(self):
        if self._note_array is None:
            self._note_array = self.performance.note_array()
        return self._note_array

    @property
    def duration(self):
        return max(self.get_all_offsets())

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

    def generate_filelist_labs(self, name, lab, pad: int) -> str:
        threshold_mask = self.oo_array.max(0)
        non_zero = np.where(threshold_mask)[0]
        zero = np.where(~threshold_mask)[0]
        res_list_pos = self._segment(name, lab, non_zero)
        res_list_neg = self._segment(name, lab, zero)
        total_no_frames = sum([j - i for _, _, i, j in res_list_pos])
        res_list_neg = self._rebalance(
            int(total_no_frames * self.percentage_negative),
            res_list_neg, min_no_frames=1)
        full_list = res_list_neg + res_list_pos
        full_list.sort(key=lambda x: (x[0], x[2]))
        # Merge potentially overlapping sections
        # This will only merge neighbouring negative and positive sections, as
        # those may have no distance between them.
#        final_list = []
#        first_item = full_list[0]
#        prev_item = full_list[0]
#        for i in full_list[1:]:
#            if i[0] == prev_item[0] and prev_item[3] - i[2] >= 0:
#                prev_item = i
#                continue
#            item = [
#                first_item[0], first_item[1], first_item[2], prev_item[3]
#            ]
#            final_list.append(item)
#            first_item = i
#            prev_item = i
#
#        if first_item != prev_item:
#            final_list.append(
#                [first_item[0], first_item[1], first_item[2], prev_item[3]])
        # Apply the padding.
        final_list = [
            (filename, lab, max(start - pad, 0),
             min(end + pad, self.oo_array.shape[1])) for
            filename, lab, start, end in full_list
        ]
        final_list = [i for i in final_list if i[3] - i[2] >= pad+1+pad]
        return "\n".join([" ".join([str(j) for j in i]) for i in final_list])

    @staticmethod
    def _rebalance(max_frames: float, old_list: list,
                   min_no_frames: int) -> list:
        new_list = []
        total_frames = 0
        while total_frames < max_frames:
            possible_choices = [
                i for i in old_list if i not in new_list and
                                       i[3] - i[2] >= min_no_frames
            ]
            if not possible_choices:
                break
            new_element = choice(possible_choices)
            total_frames += new_element[3] - new_element[2]
            new_list.append(list(new_element))

        # If we've added too much, we now shrink the sections
        while total_frames > (max_frames+min_no_frames):
            possible_idxs = []
            for i in range(len(new_list)):
                if new_list[i][3] - new_list[i][2] > min_no_frames:
                    possible_idxs.append(i)
            if choice([True, False]):
                new_list[choice(possible_idxs)][2] += 1
            else:
                new_list[choice(possible_idxs)][3] -= 1
            total_frames -= 1
        return new_list

    def _segment(self, name, lab, mask):
        segments = []
        segment_size = 1
        for i in range(1, len(mask)):
            if mask[i-1] == mask[i] + 1:
                segment_size += 1
                print("Hit")
                continue
            segments.append((name, lab, mask[i-segment_size],
                             mask[i-1]+1))
            segment_size = 1
        return segments

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
        start = int(timestamps[0] / (1 / self.time_div))
        end = int(timestamps[1] / (1 / self.time_div))

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
        ]) - self.piano_shift
        if any(notes > 87) or any(notes < 0):
            raise AttributeError("The note array seems to be invalid")
        return notes

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
