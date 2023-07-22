import kornia.feature
import torch
import torchvision
from torchvision.models.video.mvit import MViT_V2_S_Weights
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import autocontrast
from ppan.config import device
import matplotlib.pyplot as plt


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class PPAnVideoPreprocesser:
    LINE_COLOUR = 0.
    EDGE_THRESHOLD = 0.025

    def __init__(self):
        self.mvit_transform = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        self.resize = torchvision.transforms.Resize((512, 512),
                                                    antialias=True)
        self.nearest_resize = torchvision.transforms.Resize(
            (512, 512),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        self.bw = torchvision.transforms.Grayscale()
        self.edge_detector = kornia.feature.SOLD2_detector().to(device).eval()
        self.pil = torchvision.transforms.ToPILImage()
        self.piano_window = None
        self.downsize = torchvision.transforms.Resize((224, 224),
                                                      antialias=True)

    def identify_piano(self, frame):
        frame = torch.swapaxes(frame, 1, 2)
        frame = self.resize(frame)
        component_masks = self.segment_image(frame)
        component_boxes = masks_to_boxes(component_masks).type(torch.long)
        piano_window = self.find_piano(frame, component_boxes)

        return piano_window

    def find_piano(self, frame, object_boxes):
        valid = []
        equal_ratio_list = []
        centers: list[tuple[torch.Tensor, torch.Tensor]] = []
        frame = autocontrast(frame)
        for i in range(object_boxes.shape[0]):
            box = object_boxes[i, :]
            box_size = (box[0] - box[2]) * (box[1] - box[3])
            # cut out large boxes
            if box_size > 200*200 or box_size < 15:
                continue
            # Get the actual image that the box represents
            box_img = frame[:, box[1]:box[3], box[0]:box[2]]

            if not self._check_color(box_img):
                continue

            # If there is a black key, there should also be white keys
            bw_box = self.bw(box_img)
            bw_box_contrast = autocontrast(bw_box)
            thresholded_contrast = torch.where(bw_box_contrast > 150,
                                               0, 1)
            thresholded = torch.where(bw_box > 30,
                                      0, 1)
            counts_cont = torch.bincount(torch.flatten(thresholded_contrast),
                                        minlength=2)
            counts = torch.bincount(torch.flatten(thresholded),
                                    minlength=2)
            color_ratio = counts[0] / counts[1]
            color_ratio_contrast = counts_cont[0] / counts_cont[1]
            equal_ratio = torch.isclose(color_ratio_contrast,
                                        torch.tensor(.7), .4)
            if not equal_ratio and \
                    color_ratio < 5:
                continue
            if equal_ratio:
                equal_ratio_list.append(i)
            else:
                valid.append(i)

        # Boxes must be close to at least one other box.
        invalid = []
        for i in equal_ratio_list:
            dist = []
            for j in equal_ratio_list:
                dist.append(self._bb_edge_distance(object_boxes[i, :],
                                                   object_boxes[j, :]))
            if not [j for j in dist if 0.1 < j < 10]:
                invalid.append(i)

        [equal_ratio_list.remove(i) for i in invalid]

        close_boxes = []
        for i in equal_ratio_list:
            for j in valid:
                dist = self._bb_edge_distance(object_boxes[i, :],
                                              object_boxes[j, :])
                if dist < 10:
                    if j not in close_boxes:
                        close_boxes.append(j)

        if not equal_ratio_list or not close_boxes:
            return

        black_keys = object_boxes[equal_ratio_list, :]
        close_boxes = object_boxes[close_boxes, :]

        box = self._check_box_ratios(black_keys, close_boxes, frame)
        """
        drawn = torchvision.utils.draw_bounding_boxes(frame,
                                                      best_box[0])
        drawn = torch.swapaxes(drawn, 0, 2)
        plt.figure()
        plt.imshow(drawn)
        plt.show()       
        """
        return box

    @staticmethod
    def _check_color(img):
        # The only colors present should be black or white
        diff = torch.diff(img.type(torch.float), dim=0)
        diff = torch.abs(torch.diff(diff, dim=0))
        if torch.any(diff > 50):
            return False
        return True

    def _check_box_ratios(self, black_keys, boxes, frame,
                          ratio: float = 0.127):
        """
        Generate all possible combinations of boxes and return the one closest
        to the given x:y ratio.
        """
        combined = torch.concat((black_keys, boxes), dim=0)
        best_box = None
        best_loss = torch.inf
        generated = []
        for i in range(black_keys.shape[0]):
            for j in range(combined.shape[0]):
                x0 = black_keys[i, 0]
                y0 = black_keys[i, 1]
                y1 = combined[j, 3]
                x1 = combined[j, 2]
                if x1 < x0 or y1 < y0:
                    continue
                box = torch.tensor([x0, y0, x1, y1])
                generated.append(box)
                r = (x1 - x0) / (y1 - y0)
                loss = abs(ratio - r)
                if loss < best_loss:
                    if (x1 - x0) * (y1 - y0) > 100*30:
                        box_img = frame[:, box[1]:box[3], box[0]:box[2]]
                        if self._check_color(box_img):
                            best_loss = loss
                            best_box = torch.tensor([x0, y0, x1, y1])[None, :]

        return best_box

    @staticmethod
    def _bb_edge_distance(b1, b2):
        """
        Computes minimum distance between the edges of two bounding boxes.
        See here:
        https://stackoverflow.com/questions/4978323/how-to-calculate-distance-between-two-rectangles-context-a-game-in-lua

        Returns
        -------
        distance : float
        """
        x1, y1, x1b, y1b = b1
        x2, y2, x2b, y2b = b2
        left = x2b < x1
        right = x1b < x2
        bottom = y2b < y1
        top = y1b < y2
        if top and left:
            return (torch.tensor([x1, y1b])-torch.tensor([x2b, y2])).pow(2).sum().sqrt()
        elif left and bottom:
            return (torch.tensor([x1, y1])-torch.tensor([x2b, y2b])).pow(2).sum().sqrt()
        elif bottom and right:
            return (torch.tensor([x1b, y1])-torch.tensor([x2, y2b])).pow(2).sum().sqrt()
        elif right and top:
            return (torch.tensor([x1b, y1b])-torch.tensor([x2, y2])).pow(2).sum().sqrt()
        elif left:
            return x1 - x2b
        elif right:
            return x2 - x1b
        elif bottom:
            return y1 - y2b
        elif top:
            return y2 - y1b
        else:             # rectangles intersect
            return 0.

    def segment_image(self, img):
        """
        Segment an image according to its edges.

        Parameters
        ----------
        img : torch.Tensor

        Returns
        -------
        components : torch.Tensor
        """
        img = self.bw(img)
        img = torch.unsqueeze(img, 0)
        img = img / 255
        edges = self.edge_detector(img)
        lines = torch.where(edges['line_heatmap'] > self.EDGE_THRESHOLD,
                            0., 1.)
        components = kornia.contrib.connected_components(lines,
                                                         num_iterations=1000)
        classes = torch.unique(components)
        # Remove the edges as they're not needed
        classes = classes[classes != 0.]
        return components == classes[:, None, None]

    def generate_piano_crop(self, video):
        """
        Take a video and generate a crop around the piano.

        Parameters
        ----------
        video : List
        """
        self.piano_window = None
        windows = []
        for i in video:
            window = self.identify_piano(i)
            if window is not None:
                windows.append(torch.squeeze(window, dim=0))
        if not windows:
            return
        scores = []
        for i in windows:
            y_len = i[3] - i[1]
            x_len = i[2] - i[0]
            ratio_score = abs((x_len / y_len) - 0.127)
            scores.append(ratio_score)
        ideal_window = torch.argmin(torch.tensor(scores))
        """
        drawn = torchvision.utils.draw_bounding_boxes(self.resize(video[0]),
                                                      torch.stack(windows))
        drawn = torch.swapaxes(drawn, 0, 2)
        plt.figure()
        plt.imshow(drawn)
        plt.show()
        """
        if scores[ideal_window] > 0.1:
            return
        self.piano_window = windows[ideal_window]

    def frame_transform(self, frame: torch.Tensor, pad: int = None):
        """
        Preprocessing to be done on a per frame basis.

        Parameters
        ----------
        frame : torch.Tensor
        pad : int
            Amount to add to edges, defaults to 20

        Returns
        -------
        frame : torch.Tensor
        """
        if self.piano_window is None:
            return self.downsize(frame)

        if pad is None:
            pad = 20
        window = torch.clone(self.piano_window)
        frame_resized = self.resize(frame)
        window = torch.squeeze(window)
        window[0] -= pad
        window[1] -= pad
        window[2] += pad
        window[3] += pad
        frame_cropped = frame_resized[:, window[1]:window[3],
                                      window[0]:window[2]]
        if 0 in frame_cropped.shape:
            return self.downsize(frame)
        return self.downsize(frame_cropped)

    def video_transform(self, video: torch.Tensor):
        """
        Preprocessing to be done on a per video basis.

        Parameters
        ----------
        video : torch.Tensor

        Returns
        -------
        preprocessed_video : torch.Tensor
        """
        return self.mvit_transform(video)
