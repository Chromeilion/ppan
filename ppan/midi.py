from miditoolkit.pianoroll import parser as pr_parser
import numpy as np
import cv2
import torch


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
