import os

import cv2
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

GAUSSIAN_VALUE = np.array([[0.0111, 0.0388, 0.0821, 0.1054, 0.0821, 0.0388, 0.0111],
                           [0.0388, 0.1353, 0.2865, 0.3679, 0.2865, 0.1353, 0.0388],
                           [0.0821, 0.2865, 0.6065, 0.7788, 0.6065, 0.2865, 0.0821],
                           [0.1054, 0.3679, 0.7788, 1.0000, 0.7788, 0.3679, 0.1054],
                           [0.0821, 0.2865, 0.6065, 0.7788, 0.6065, 0.2865, 0.0821],
                           [0.0388, 0.1353, 0.2865, 0.3679, 0.2865, 0.1353, 0.0388],
                           [0.0111, 0.0388, 0.0821, 0.1054, 0.0821, 0.0388, 0.0111]])


class VisionParkingSlotDataset(Dataset):
    def __init__(self, image_path, label_path, data_size, resolution):
        self.length = data_size
        self.image_list = []
        self.label_list = []
        index = 0
        for item_name in os.listdir(image_path):
            item_label = loadmat("%s%s.mat" % (label_path, item_name[:-4]))
            slots = item_label['slots']
            if len(slots) > 0:
                item_image = cv2.resize(cv2.imread(image_path + item_name), (resolution, resolution))
                item_image = np.transpose(item_image, (2, 0, 1))
                self.image_list.append(item_image)

                marks = item_label['marks']
                mark_label = self._get_mark_label(marks, slots, resolution)
                slot_label = np.zeros([3, resolution, resolution])
                for mark in mark_label:
                    slot_label[0, mark[1] - 3:mark[1] + 4, mark[0] - 3:mark[0] + 4] += GAUSSIAN_VALUE
                    slot_label[1, mark[1] - 3:mark[1] + 4, mark[0] - 3:mark[0] + 4] += mark[2]
                    slot_label[2, mark[1] - 3:mark[1] + 4, mark[0] - 3:mark[0] + 4] += mark[3]
                self.label_list.append(slot_label)

                index += 1
                if index == data_size:
                    break

    @staticmethod
    def _get_mark_label(marks, slots, resolution):
        """
        :param marks:       x, y
        :param slots:       pt1, pt2, _, rotate_angle
        :param resolution:  224x224
        :return:
        """
        temp_mark_label = []
        for mark in marks:
            mark_x_re, mark_y_re = mark[0] * resolution / 600, mark[1] * resolution / 600
            temp_mark_label.append([mark_x_re, mark_y_re, [], []])

        for slot in slots:
            mark_vector = marks[slot[1] - 1] - marks[slot[0] - 1]
            mark_vector_length = np.sqrt(mark_vector[0] ** 2 + mark_vector[1] ** 2)
            if mark_vector[0] > 0:
                mark_direction = np.arcsin(mark_vector[1] / mark_vector_length)
            else:
                mark_direction = np.pi - np.arcsin(mark_vector[1] / mark_vector_length)
            slot_direction = mark_direction - slot[3] * np.pi / 180
            slot_cos = np.cos(slot_direction)
            slot_sin = np.sin(slot_direction)

            temp_mark_label[slot[0] - 1][2].append(slot_cos)
            temp_mark_label[slot[0] - 1][3].append(slot_sin)
            temp_mark_label[slot[1] - 1][2].append(slot_cos)
            temp_mark_label[slot[1] - 1][3].append(slot_sin)

        mark_label = []
        for mark in temp_mark_label:
            if len(mark[2]) > 0:
                mark_cos = np.mean(mark[2])
                mark_sin = np.mean(mark[3])
                mark_angle_base = np.sqrt(mark_cos ** 2 + mark_sin ** 2)
                mark_cos = mark_cos / mark_angle_base
                mark_sin = mark_sin / mark_angle_base
                mark_label.append([int(round(mark[0])), int(round(mark[1])), mark_cos, mark_sin])
        return mark_label

    def __getitem__(self, item):
        return self.image_list[item], self.label_list[item]

    def __len__(self):
        return self.length
