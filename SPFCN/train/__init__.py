import os
import dill
from datetime import datetime

import torch

from .trainer import Trainer
from .validator import Validator


@torch.no_grad()
def auto_validate(dataset,
                  network,
                  device_id: int = 0):
    device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)
    network.eval()
    auto_validator = Validator(dataset, network, device)
    auto_validator.step()
    auto_validator.get_network_inference_time()
    auto_validator.get_detector_inference_time()


def auto_train(dataset,
               valid_dataset,
               network,
               device_id: int = 0,
               load_path: str = None,
               save_path: str = None,
               epoch_limit: int = 1000,
               lr: float = 1e-3):
    assert epoch_limit > 0 and lr > 0

    device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)

    if load_path is None:
        network.initialize_weights()
    else:
        network.load_state_dict(torch.load(load_path, map_location=device))
    network.train()

    if save_path is None:
        save_path = './'
    elif save_path[-1] != '/':
        save_path += '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    epoch_unit = epoch_limit // 10
    stage = "warm_up"
    auto_trainer = Trainer(dataset, network, device, lr)
    
    for epoch in range(1, epoch_limit + 1):
        print("Train model ... ")
        if epoch < epoch_unit:
            pass
        elif epoch == epoch_unit:
            auto_trainer.update_cost(neg=0.1)
            stage = "auto_select"
        elif epoch < 3 * epoch_unit:
            network.auto_select()
        elif epoch == 3 * epoch_unit:
            network.enforce_select()
            stage = "enforce_select"
        elif epoch < 5 * epoch_unit:
            stage = "prune_channel"
        elif epoch < 8 * epoch_unit:
            network.prune_channel()
        elif epoch == 8 * epoch_unit:
            auto_trainer.with_regularization = False
            stage = "fine_tuning"

        epoch_time, epoch_loss = auto_trainer.step()
        if epoch == epoch_limit:
            network.merge()
            stage = "merge_bn"

        torch.save(network.state_dict(), "%s%s_epoch%d_loss%d.pkl" % (save_path, stage, epoch, int(epoch_loss)))
        torch.save(dill.dumps(network), "%s%s_epoch%d_loss%d.pt" % (save_path, stage, epoch, int(epoch_loss)))

        curr = datetime.now()
        info = '{:02d}:{:02d}:{:02d} '.format(curr.hour, curr.minute, curr.minute)
        info += 'Epoch: {}/{}[{}], Loss: {:.3f}, '.format(epoch, epoch_limit, stage, epoch_loss)
        time_left = (epoch_limit - epoch) * epoch_time
        info += 'Time left: ' + 'About {} minutes'.format(int(time_left / 60)) if time_left < 3600 \
            else 'About {} hours'.format(int(time_left / 3600)) if time_left < 36000 else 'Just go to sleep'
        print(info)
        
        print("Validate model ... ")
        auto_validate(valid_dataset, network, device_id)

    print(network.get_encoder())
