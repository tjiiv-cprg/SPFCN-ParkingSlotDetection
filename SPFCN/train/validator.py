import torch
from .loss import FocalLossWithTrigonometric


class Validator(object):
    def __init__(self, valid_dataset, network, device, lr):
        self.valid_dataset = valid_dataset
        self.network = network.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.cost_function = FocalLossWithTrigonometric(neg=0.01).to(device)
        self.with_regularization = True

    def set_network(self, network):
        self.network = network.to(self.device)

    def step(self):
        self.valid_dataset.refresh()
        validating_images, validating_labels = self.valid_dataset.next()
        valid_sum_loss = 0
        valid_index = 1
        with torch.no_grad():
            while validating_images is not None and validating_labels is not None:
                valid_mark, valid_direction = self.network(validating_images)

                valid_step_loss = self.cost_function(valid_mark, valid_direction, validating_labels)
                valid_info = "\tValidation - Index:{}, Step loss:{:.3f}".format(valid_index, valid_step_loss.item())

                if self.with_regularization:
                    valid_regularization = self.network.get_regularization()
                    valid_step_loss += valid_regularization
                    valid_info += ", Regularization:{:.3f}".format(valid_regularization.item())

                valid_sum_loss += valid_step_loss.item()

                valid_index += 1
                validating_images, validating_labels = self.valid_dataset.next()

        return valid_sum_loss / valid_index
