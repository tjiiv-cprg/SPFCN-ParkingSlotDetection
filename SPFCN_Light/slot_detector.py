"""
    slot_detector.py
        The model of the parking slot detection.

    Example:
        model_parameter_path = "..."
        detector = Detector(model_parameter_path, device_id)
        inference_result = detector(inference_image)

    Data form:
        inference_image: Gray Image, numpy.array with Size([224, 224])
        inference_result: list of slot_points

    Requirements:
        pytorch == 1.2.0
        numpy == 1.16.5
        opencv == 3.4.2

    2020.09.15
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
    def __init__(self, file_name, device_id=0):
        super().__init__()
        self.device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)
        self._network = Hourglass([1, 40, 56, 55, 60, 59, 61, 59, 60, 55, 56, 40, 3]).to(self.device)
        self._network.load_state_dict(torch.load(file_name), strict=True)
        print("Success load file {}.".format(file_name))
        self._network.eval()

        self.temp_h = torch.ones((1, 224)).to(self.device)
        self.temp_w = torch.ones((224, 1)).to(self.device)

    @torch.no_grad()
    def __call__(self, inference_image):
        inference_image = torch.from_numpy(inference_image).to(self.device).float()
        inference_image = ((inference_image - torch.mean(inference_image)) / torch.std(inference_image))

        outputs = self._network(inference_image.unsqueeze_(dim=0).unsqueeze_(dim=0))[0]

        output = outputs[0]
        left = torch.cat((output[1:, :], self.temp_h), dim=0)
        right = torch.cat((self.temp_h, output[:-1, :]), dim=0)
        up = torch.cat((output[:, 1:], self.temp_w), dim=1)
        down = torch.cat((self.temp_w, output[:, :-1]), dim=1)
        mask = (output > 0) * (output > left) * (output > right) * (output > up) * (output > down)
        mask, mask_points, mask_point_count = mask.float(), torch.where(mask), mask.int().sum().item()

        entry = outputs[1].sigmoid_()
        side = outputs[2].sigmoid_()

        result = []
        for mask_point_index1 in range(mask_point_count - 1):
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
                if distance < 56 or 72 < distance < 128 or 144 < distance:
                    continue

                if y1 < y2:
                    y_min = y1 - 1
                    y_max = y2 + 2
                    canvas = np.zeros([x2 - x1 + 3, y2 - y1 + 3])
                    line(canvas, (1, 1), (y2 - y1 + 1, x2 - x1 + 1), 1, thickness=1)
                else:
                    y_min = y2 - 1
                    y_max = y1 + 2
                    canvas = np.zeros([x2 - x1 + 3, y1 - y2 + 3])
                    line(canvas, (1, x2 - x1 + 1), (y1 - y2 + 1, 1), 1, thickness=1)
                canvas = torch.from_numpy(canvas).float().to(self.device)
                if (canvas * mask[x1 - 1:x2 + 2, y_min:y_max]).sum() != 2:
                    continue

                score_entry = (canvas * entry[x1 - 1:x2 + 2, y_min:y_max]).sum() / distance
                if score_entry < 0.25:
                    continue

                direct = (side[x1 - 2:x1 + 3, y1 - 2:y1]).sum() > (side[x1 - 2:x1 + 3, y1:y1 + 3]).sum()
                if direct:
                    direct_vector_y = 16 * (x1 - x2) / distance
                    direct_vector_x = -16 * (y1 - y2) / distance
                else:
                    direct_vector_y = -16 * (x1 - x2) / distance
                    direct_vector_x = 16 * (y1 - y2) / distance

                vec_x = int(direct_vector_x)
                vec_y = int(direct_vector_y)
                side_score1 = side[x1 + vec_x - 2:x1 + vec_x + 3, y1 + vec_y - 2:y1 + vec_y + 3].sum()
                if side_score1 < 0.25:
                    continue

                side_score2 = side[x2 + vec_x - 2:x2 + vec_x + 3, y2 + vec_y - 2:y2 + vec_y + 3].sum()
                if side_score2 < 0.25:
                    continue

                if score_entry * (side_score1 + side_score2) < 1:
                    continue

                pt0 = (y1, x1)
                pt1 = (y2, x2)
                pt2 = (y1 + direct_vector_y * 6, x1 + direct_vector_x * 6)
                pt3 = (y2 + direct_vector_y * 6, x2 + direct_vector_x * 6)

                if direct:
                    result.append((pt0, pt1, pt2, pt3))
                else:
                    result.append((pt1, pt0, pt3, pt2))
        return result
