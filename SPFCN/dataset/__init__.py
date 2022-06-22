import torch
from torch.utils.data import DataLoader

from .dataset import VisionParkingSlotDataset
from .prefetcher import DataPrefetcher


def get_training_set(data_size: int,
                     batch_size: int,
                     resolution: int = 224,
                     device_id: int = 0):
    assert 0 < data_size < 6596 and 0 < batch_size and 0 < resolution

    vps_set = VisionParkingSlotDataset(
        image_path="./data/training/image/",
        label_path="./data/training/label/",
        data_size=data_size,
        resolution=resolution)

    return DataLoader(dataset=vps_set, shuffle=True, batch_size=batch_size, num_workers=4)
    # if device_id < 0:
    #     return DataLoader(dataset=vps_set, shuffle=True, batch_size=batch_size, num_workers=4)
    # else:
    #     return DataPrefetcher(device=torch.device('cuda:%d' % device_id),
    #                           dataset=vps_set, batch_size=batch_size, shuffle=True)


def get_validating_set(data_size: int,
                       batch_size: int,
                       resolution: int = 224,
                       device_id: int = 0):
    assert 0 < data_size < 1538 and 0 < batch_size and 0 < resolution
    vps_set = VisionParkingSlotDataset(
        image_path="./data/testing/image/",
        label_path="./data/testing/label/",
        data_size=data_size,
        resolution=resolution)
    if device_id < 0:
        return DataLoader(dataset=vps_set, shuffle=True, batch_size=batch_size, num_workers=4)
    else:
        return DataPrefetcher(device=torch.device('cuda:%d' % device_id),
                              dataset=vps_set, batch_size=batch_size, shuffle=False)


# TODO
def get_testing_set(data_size: int,
                       batch_size: int,
                       resolution: int = 224,
                       device_id: int = 0):
    assert 0 < data_size < 1538 and 0 < batch_size and 0 < resolution
    vps_set = VisionParkingSlotDataset(
        image_path="./data/testing/image/",
        label_path="./data/testing/label/",
        data_size=data_size,
        resolution=resolution)
    if device_id < 0:
        return DataLoader(dataset=vps_set, shuffle=True, batch_size=batch_size, num_workers=4)
    else:
        return DataPrefetcher(device=torch.device('cuda:%d' % device_id),
                              dataset=vps_set, batch_size=batch_size, shuffle=False)


