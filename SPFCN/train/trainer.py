from time import time

import torch

from .loss import FocalLossWithTrigonometric


class Trainer(object):
    def __init__(self, dataset, network, device, lr):
        self.dataset = dataset
        self.network = network.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.cost_function = FocalLossWithTrigonometric(neg=0.01).to(device)
        self.with_regularization = True

    def step(self):
        timestamp = time()
        self.dataset.refresh()
        training_images, training_labels = self.dataset.next()
        index = 1
        sum_loss = 0
        while training_images is not None and training_labels is not None:
            mark, direction = self.network(training_images)

            step_loss = self.cost_function(mark, direction, training_labels)
            info = "\tIndex:{}, Step loss:{:.3f}".format(index, step_loss.item())

            if self.with_regularization:
                regularization = self.network.get_regularization()
                step_loss += regularization
                info += ", Regularization:{:.3f}".format(regularization.item())

            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
            sum_loss += step_loss.item()

            print(info)
            index += 1
            training_images, training_labels = self.dataset.next()

        return time() - timestamp, sum_loss / index

    def update_cost(self, **kwargs):
        self.cost_function = FocalLossWithTrigonometric(**kwargs).to(self.device)
