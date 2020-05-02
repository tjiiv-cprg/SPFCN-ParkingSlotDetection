"""
    slot_detector.py
        The model of the parking slot detection.

    Example:
        model_parameter_path = "..."
        detector = Detector(model_parameter_path)
        inference_result = detector(inference_image)

    Data form:
        inference_image: Gray Image, torch.Tensor with Size([224, 224])
        inference_result: list of slot_points

    Requirements:
        pytorch == 1.2.0
        numpy == 1.16.5
        opencv == 3.4.2

    2019.11.28
"""

from cv2 import line
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class ConvMode(nn.Module):
    def __init__(self, mode, ipc, opc, sel_index, stride=1, only_conv=False):
        super().__init__()
        if mode == 0:
            self._conv = nn.Conv2d(ipc, opc, 3, stride, padding=sel_index, dilation=sel_index, bias=False)
        elif mode == 1:
            self._conv = nn.Conv2d(ipc, opc, 2 * sel_index + 1, stride, padding=sel_index, dilation=1, bias=False)
        self._bn = nn.Sequential() if only_conv else nn.BatchNorm2d(opc)
        self._relu = nn.Sequential() if only_conv else nn.ReLU(inplace=True)

    def forward(self, feature):
        return self._relu(self._bn(self._conv(feature)))


class Hourglass(nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        mode_list = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        dilation_list = [2, 2, 1, 3, 1, 1, 2, 1, 3, 1, 2, 1]
        stride_list = [1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]

        self._conv_trunk = nn.ModuleList()
        for depth_index in range(11):
            self._conv_trunk.append(ConvMode(
                mode_list[depth_index], channel_list[depth_index], channel_list[depth_index + 1],
                dilation_list[depth_index], stride_list[depth_index]))
        self._conv_trunk.append(ConvMode(
            mode_list[-1], channel_list[-2], channel_list[-1],
            dilation_list[-1], stride_list[-1], only_conv=True))

    def forward(self, feature):
        skip = []
        for index, conv in enumerate(self._conv_trunk):
            if 1 <= index <= 5:
                skip.append(feature)
            elif 7 <= index <= 11:
                feature = f.interpolate(feature, scale_factor=2) + skip.pop()
            feature = conv(feature)
        return feature


class Detector(object):
    def __init__(self, file_name):
        super().__init__()
        self._network = Hourglass([1, 40, 56, 55, 60, 59, 61, 59, 60, 55, 56, 40, 3]).cuda()
        self._network.load_state_dict(torch.load(file_name), strict=True)
        print("Success load file {}.".format(file_name))
        self._network.eval()

    @staticmethod
    def _mask_detection(output, threshold=0):
        temp_h = torch.ones((1, 224)).cuda()
        temp_w = torch.ones((224, 1)).cuda()

        left = torch.cat((output[1:, :], temp_h), dim=0)
        right = torch.cat((temp_h, output[:-1, :]), dim=0)
        up = torch.cat((output[:, 1:], temp_w), dim=1)
        down = torch.cat((temp_w, output[:, :-1]), dim=1)

        mask = (output > threshold) * (output > left) * (output > right) * (output > up) * (output > down)
        return mask.float(), torch.where(mask), mask.int().sum().item()

    def __call__(self, inference_image, threshold=0.1):
        inference_image = ((inference_image - torch.mean(inference_image))
                           / torch.std(inference_image)).unsqueeze_(dim=0).unsqueeze_(dim=0)
        with torch.no_grad():
            outputs = self._network(inference_image)
            mask, mask_points, mask_point_count = self._mask_detection(outputs[0, 0])
            entry = outputs[0, 1].sigmoid_()
            side = outputs[0, 2].sigmoid_()

            result = []
            for mask_point_index1 in range(mask_point_count):
                x1 = mask_points[0][mask_point_index1].item()
                y1 = mask_points[1][mask_point_index1].item()
                if x1 < 16 or 208 < x1 or y1 < 16 or 208 < y1:
                    continue

                for mask_point_index2 in range(mask_point_index1 + 1, mask_point_count):
                    x2 = mask_points[0][mask_point_index2].item()
                    y2 = mask_points[1][mask_point_index2].item()

                    if x2 < 16 or 208 < x2 or y2 < 16 or 208 < y2:
                        continue

                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # print("distance", distance)
                    if distance < 56 or 72 < distance < 128 or 144 < distance:
                        continue

                    if y1 < y2:
                        y_min = y1 - 1
                        y_max = y2 + 2
                        canvas = np.zeros([x2 - x1 + 3, y2 - y1 + 3])
                        line(canvas, (1, 1), (y2 - y1 + 1, x2 - x1 + 1), 1, 2)
                    else:
                        y_min = y2 - 1
                        y_max = y1 + 2
                        canvas = np.zeros([x2 - x1 + 3, y1 - y2 + 3])
                        line(canvas, (1, x2 - x1 + 1), (y1 - y2 + 1, 1), 1, 2)
                    canvas = torch.from_numpy(canvas).float().cuda()
                    if canvas.shape != torch.Size([x2 - x1 + 3, y_max - y_min]):
                        continue

                    if (canvas * mask[x1 - 1:x2 + 2, y_min:y_max]).sum() != 2:
                        continue

                    score_entry = (canvas * entry[x1 - 1:x2 + 2, y_min:y_max]).sum() / distance
                    # print("score_entry", score_entry.item())
                    if score_entry < threshold:
                        continue

                    direct = (side[x1 - 2:x1 + 3, y1 - 2:y1]).sum() > (side[x1 - 2:x1 + 3, y1:y1 + 3]).sum()
                    if direct:
                        direct_vector = (16 * (x1 - x2) / distance, 16 * (y2 - y1) / distance)
                    else:
                        direct_vector = (16 * (x2 - x1) / distance, 16 * (y1 - y2) / distance)

                    vec_x = int(direct_vector[1])
                    vec_y = int(direct_vector[0])
                    side_score1 = side[x1 + vec_x - 2:x1 + vec_x + 3, y1 + vec_y - 2:y1 + vec_y + 3].sum()
                    side_score2 = side[x2 + vec_x - 2:x2 + vec_x + 3, y2 + vec_y - 2:y2 + vec_y + 3].sum()
                    # print("side score", side_score1, side_score2)
                    if side_score1 < threshold or side_score2 < threshold:
                        continue

                    # print("score_final", score_entry * (side_score1 + side_score2))
                    if score_entry * (side_score1 + side_score2) < threshold * 8:
                        continue

                    pt0 = (y1, x1)
                    pt1 = (y2, x2)

                    if distance < 85:
                        pt2 = (pt0[0] + direct_vector[0] * 6, pt0[1] + direct_vector[1] * 6)
                        pt3 = (pt1[0] + direct_vector[0] * 6, pt1[1] + direct_vector[1] * 6)
                    else:
                        pt2 = (pt0[0] + direct_vector[0] * 3, pt0[1] + direct_vector[1] * 3)
                        pt3 = (pt1[0] + direct_vector[0] * 3, pt1[1] + direct_vector[1] * 3)

                    if direct:
                        result.append((pt0, pt1, pt2, pt3))
                    else:
                        result.append((pt1, pt0, pt3, pt2))
        return result
