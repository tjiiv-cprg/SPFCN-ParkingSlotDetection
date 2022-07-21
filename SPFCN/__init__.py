import torch
from torch.backends import cudnn

from .dataset import get_training_set, get_validating_set, get_testing_set
from .model.network import SlotNetwork
from .train import auto_train, auto_validate
from .test import auto_test


def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def slot_network_training(data_num, batch_size, valid_data_num, valid_batch_size, epoch, input_res,  device_id=0, num_workers=0):
    # Initial
    setup(19960229)
    net = SlotNetwork([32, 44, 64, 92, 128], device_id=device_id)
    
    # Train
    auto_train(get_training_set(data_num, batch_size, input_res, device_id, num_workers), 
               get_validating_set(valid_data_num, valid_batch_size, input_res, device_id, num_workers),
               net, device_id=device_id,
               epoch_limit=epoch, save_path="parameters/")


def slot_network_testing(parameter_path, data_num, batch_size, input_res, device_id=0):
    # Initial
    setup(19960229)
    net = SlotNetwork([32, 44, 64, 92, 128], device_id)

    # Test
    auto_test(get_testing_set(data_num, batch_size, input_res, device_id, num_workers=0), net, device_id, parameter_path)
