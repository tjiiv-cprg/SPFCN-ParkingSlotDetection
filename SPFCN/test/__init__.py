import os
import torch
import dill
from .tester import Tester


@torch.no_grad()
def auto_test(dataset,
                  network,
                  device_id: int = 0,
                  load_path: str = None):
    device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)
    
    try: 
        net_path = load_path + '.pkl'
        assert os.path.exists(net_path)
        network.load_state_dict(torch.load(net_path, map_location=device))
    except RuntimeError: 
        net_path = load_path + '.pt'
        assert os.path.exists(net_path)
        network = torch.load(net_path, map_location=device) 
        network=dill.loads(network)
    network.eval()

    auto_tester = Tester(dataset, network, device)
    auto_tester.step()
    auto_tester.get_network_inference_time()
    auto_tester.get_detector_inference_time()
