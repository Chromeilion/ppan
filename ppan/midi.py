import io
from typing import Tuple

import cv2
import mido
import numpy as np
import torch
from miditok import Structured
from miditoolkit import MidiFile
from miditoolkit.pianoroll import parser as pr_parser


class PPAnMidi:
    """
    Class for handling midi operations. Things like tokenization and loading.
    """
    def __init__(self, max_len: int):
        special_tokens = ["BOS", "EOS", "PAD"]
        self.tokenizer = Structured(pitch_range=(range(0, 127)),
                                    special_tokens=special_tokens,
                                    nb_velocities=1)
        self.vocab_len = len(self.tokenizer)
        self.max_len = max_len

    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls self.tokenize_midi
        """
        return self.tokenize_midi(*args, **kwargs)

    def tokenize_midi(self,
                      midi,
                      timestamps: Tuple[float, float]) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a section from a midi based on timestamps and tokenize it.
        This implementation is a little absurd, but It's what I came up with
        and it works.

        Parameters
        ----------
        midi : mido.MidiFile
        timestamps : Tuple[float, float]

        Returns
        -------
        tokens : Tuple[torch.Tensor, torch.Tensor]
        """
        new_midi = mido.MidiFile(type=0)
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)
        current_time = 0
        tempo = 500000
        for i in midi.tracks[0]:
            if i.type == 'set_tempo':
                tempo = i.tempo

            current_time += mido.tick2second(i.time,
                                             midi.ticks_per_beat,
                                             tempo)

            if timestamps[0] <= current_time <= timestamps[1]:
                new_track.append(i)
            elif current_time > timestamps[1]:
                break

        with io.BytesIO() as f:
            new_midi.save(file=f)
            # Make sure to go back to the start of the file after writing!
            f.seek(0)
            tokens = self.tokenizer.midi_to_tokens(MidiFile(file=f),
                                                   add_special_tokens=True)

            if len(tokens) != 0:
                tokens = list(tokens[0])

            tokens.insert(0, self.tokenizer["BOS_None"])

            if len(tokens) > self.max_len-1:
                print(len(tokens))
                tokens = tokens[:self.max_len-2]
            tokens.append(self.tokenizer["EOS_None"])

            while len(tokens) < self.max_len:
                tokens.append(self.tokenizer["PAD_None"])

        tokens = torch.tensor(tokens, dtype=torch.long)

        padding_mask = torch.squeeze(
            torch.where(tokens == self.tokenizer["PAD_None"],
                        True,
                        False))

        return tokens, padding_mask

    def tokens_to_midi(self, tokens: torch.Tensor):
        tokens = torch.unsqueeze(tokens, dim=1)
        midi_list = [self.tokenizer.tokens_to_midi(
            tokens=i.cpu().detach().numpy()) for i in
            torch.unbind(tokens, dim=0)]
        return midi_list


def compute_piano_img(midi):
    pianorolls = [
        pr_parser.notes2pianoroll(i.instruments[0].notes, resample_factor=0.05)
        for i in midi
    ]
    images = [np.expand_dims(i.T, 2) for i in pianorolls]
    for i, image in enumerate(images):
        if 0 in image.shape:
            zero_dims = []
            for dim, j in enumerate(image.shape):
                if j == 0:
                    zero_dims.append(dim)
            for j in zero_dims:
                image = np.insert(image, 0, 0, axis=j)
            images[i] = image

    resized = [cv2.resize(i.astype(np.float32), dsize=(128, 128),
                          interpolation=cv2.INTER_CUBIC)
               for i in images]

    resized = [np.expand_dims(i, 0) for i in resized]

    return torch.tensor(np.array(resized))
