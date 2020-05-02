from time import time

import cv2
import numpy as np
import torch


class Validator(object):
    def __init__(self, dataset, network, device):
        self.dataset = dataset
        self.network = network.to(device)
        self.device = device

        self.const_h = torch.ones((1, 224)).to(device)
        self.const_w = torch.ones((224, 1)).to(device)
        self.mark_threshold = 0.1
        self.direct_threshold = 0.95
        self.distance_threshold = 40
        self.elliptic_coefficient = 1.6

        self.mgt_threshold = 4.6
        self.iou_threshold = 0.95

    def step(self):
        self.dataset.refresh()
        validating_image, validating_label = self.dataset.next()
        index = 0
        mark_gt_count, mark_re_count, mark_co_count = 0, 0, 0
        slot_gt_count, slot_re_count, slot_co_count = 0, 0, 0
        while validating_image is not None and validating_label is not None:
            gt_mark = validating_label[0:1, 0:1]
            gt_direction = validating_label[0:1, 1:]
            gt_mark_count, gt_mark_map, gt_slot_count, gt_slot_list = \
                self.slot_detect(gt_mark[0, 0], gt_direction, True)
            mark_gt_count += gt_mark_count
            slot_gt_count += gt_slot_count

            re_mark, re_direction = self.network(validating_image)
            re_mark_count, re_mark_map, re_slot_count, re_slot_list = \
                self.slot_detect(re_mark[0, 0], re_direction, False)
            mark_re_count += re_mark_count
            slot_re_count += re_slot_count

            for ind in range(re_mark_count):
                re_x = int(re_mark_map[ind, 0])
                re_y = int(re_mark_map[ind, 1])
                angle = re_mark_map[ind, 2] * gt_direction[0, 0, re_x, re_y]
                angle += re_mark_map[ind, 3] * gt_direction[0, 1, re_x, re_y]
                distance = gt_mark[0, 0, re_x - 1:re_x + 2, re_y - 1:re_y + 2].sum()
                if angle > self.direct_threshold and distance > self.mgt_threshold:
                    mark_co_count += 1

            for ind in range(re_slot_count):
                re_pt = re_slot_list[ind]
                for jnd in range(gt_slot_count):
                    gt_pt = gt_slot_list[jnd]
                    mask_gt = cv2.fillConvexPoly(np.zeros([224, 224], dtype="uint8"), np.array(gt_pt), 1)
                    mask_re = cv2.fillConvexPoly(np.zeros([224, 224], dtype="uint8"), np.array(re_pt), 1)
                    count_and = np.sum(cv2.bitwise_and(mask_re, mask_gt))
                    count_or = np.sum(cv2.bitwise_or(mask_re, mask_gt))
                    if count_and > self.iou_threshold * count_or:
                        slot_co_count += 1

            validating_image, validating_label = self.dataset.next()
            index += 1
            print("\rIndex: {}, Mark: Precision {:.4f}, Recall {:.4f}, Slot: Precision {:.4f}, Recall {:.4f}"
                  .format(index, mark_co_count / mark_re_count, mark_co_count / mark_gt_count,
                          slot_co_count / slot_re_count, slot_co_count / slot_gt_count), end='')
        print('\r' + ' ' * 50, end="")
        print("Mark: Precision {:.4f}, Recall {:.4f}, Slot: Precision {:.4f}, Recall {:.4f}"
              .format(mark_co_count / mark_re_count, mark_co_count / mark_gt_count,
                      slot_co_count / slot_re_count, slot_co_count / slot_gt_count))

    def get_network_inference_time(self):
        def foo(img):
            _, _ = self.network(img)

        print('\rNetwork ' + self.get_inference_time(foo))

    def get_detector_inference_time(self):
        def foo(img):
            mark, direction = self.network(img)
            self.slot_detect(mark[0, 0], direction, False)

        print('\rDetector ' + self.get_inference_time(foo))

    def slot_detect(self, mark, direction, gt=False):
        # Mark detection
        if gt:
            mark_prediction = torch.nonzero(mark == 1)
        else:
            mark_prediction = torch.nonzero((mark > self.mark_threshold) *
                                            (mark > torch.cat((mark[1:, :], self.const_h), dim=0)) *
                                            (mark > torch.cat((self.const_h, mark[:-1, :]), dim=0)) *
                                            (mark > torch.cat((mark[:, 1:], self.const_w), dim=1)) *
                                            (mark > torch.cat((self.const_w, mark[:, :-1]), dim=1)))

        mark_count = len(mark_prediction)
        mark_map = torch.zeros([mark_count, 4]).to(self.device)
        mark_map[:, 0:2] = mark_prediction
        for item in mark_map:
            item[2:] = direction[0, :, item[0].int(), item[1].int()]

        # Distance map generate
        distance_map = torch.zeros([mark_count, mark_count]).to(self.device)
        for i in range(0, mark_count - 1):
            for j in range(i + 1, mark_count):
                if mark_map[i, 2] * mark_map[j, 2] + mark_map[i, 3] * mark_map[j, 3] > self.direct_threshold:
                    distance = torch.pow(torch.pow(mark_map[i, 0] - mark_map[j, 0], 2) +
                                         torch.pow(mark_map[i, 1] - mark_map[j, 1], 2), 0.5)
                    distance_map[i, j] = distance
                    distance_map[j, i] = distance

        # Slot check
        slot_list = []
        for i in range(0, mark_count - 1):
            for j in range(i + 1, mark_count):
                distance = distance_map[i, j]
                if distance > self.distance_threshold and \
                        (distance_map[i] + distance_map[j] < self.elliptic_coefficient * distance).sum() == 2:
                    slot_length = 120 if distance < 80 else 60
                    vx = torch.abs(mark_map[i, 0] - mark_map[j, 0]) / distance
                    vy = torch.abs(mark_map[i, 1] - mark_map[j, 1]) / distance
                    delta_x = -slot_length * vx if mark_map[i, 2] < 0 else slot_length * vx
                    delta_y = -slot_length * vy if mark_map[i, 3] < 0 else slot_length * vy

                    slot_list.append(((int(mark_map[i, 1]), int(mark_map[i, 0])),
                                      (int(mark_map[j, 1]), int(mark_map[j, 0])),
                                      (int(mark_map[j, 1] + delta_x), int(mark_map[j, 0] + delta_y)),
                                      (int(mark_map[i, 1] + delta_x), int(mark_map[i, 0] + delta_y))))
                    break

        return mark_count, mark_map, len(slot_list), slot_list

    def get_inference_time(self, foo):
        self.dataset.refresh()
        validating_image, _ = self.dataset.next()
        foo(validating_image)
        index = 0
        time_step = 0
        while validating_image is not None:
            timestamp = time()
            foo(validating_image)
            time_step += time() - timestamp
            validating_image, _ = self.dataset.next()
            index += 1
            print("\rIndex: {}, Inference Time: {:.1f}ms".format(index, 1e3 * time_step / index), end="")
        print('\r' + ' ' * 40, end="")
        return "Inference Time: {:.1f}ms".format(1e3 * time_step / index)
