import torch
import dill

from .network import SlotNetwork


class SlotDetector(object):
    def __init__(self, device_id: int, **kwargs):
        self.device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)
        self.config = self.update_config(**kwargs)
        self.network = SlotNetwork(self.config['dim_encoder'], device_id)
        self.network.merge()
        try: 
            self.network.load_state_dict(torch.load(self.config['parameter_path'], map_location=self.device))
        except RuntimeError:
            net_path = self.config['parameter_path'].replace('.pkl', '.pt')
            network = torch.load(net_path, map_location=self.device) 
            self.network= dill.loads(network)
        self.network.eval()

    def update_config(self, **kwargs):
        config = {
            'dim_encoder': [64, 108, 180, 304, 512],
            'parameter_path': 'l5c512_stable.pkl',
            # Inference parameters
            'const_h': torch.ones((1, 224)).to(self.device),
            'const_w': torch.ones((224, 1)).to(self.device),
            'mark_threshold': 0.25,
            'direct_threshold': 0.95,
            'distance_threshold': 40,
            'elliptic_coefficient': 1.6,
        }
        for key, value in kwargs.items():
            if key in config.keys():
                config[key] = value
        return config

    @torch.no_grad()
    def __call__(self, bev_image):
        bev_image = torch.from_numpy(bev_image).to(self.device).permute(2, 0, 1).float()
        mark, direction = self.network(bev_image.unsqueeze_(dim=0).contiguous())

        # Mark detection
        mark = mark[0, 0]
        mark_prediction = torch.nonzero((mark > self.config['mark_threshold']) *
                                        (mark > torch.cat((mark[1:, :], self.config['const_h']), dim=0)) *
                                        (mark > torch.cat((self.config['const_h'], mark[:-1, :]), dim=0)) *
                                        (mark > torch.cat((mark[:, 1:], self.config['const_w']), dim=1)) *
                                        (mark > torch.cat((self.config['const_w'], mark[:, :-1]), dim=1)))
        mark_count = len(mark_prediction)
        mark_map = torch.zeros([mark_count, 4]).to(self.device)
        mark_map[:, 0:2] = mark_prediction
        for item in mark_map:
            item[2:] = direction[0, :, item[0].int(), item[1].int()]

        # Distance map generate
        distance_map = torch.zeros([mark_count, mark_count]).to(self.device)
        for i in range(0, mark_count - 1):
            for j in range(i + 1, mark_count):
                if mark_map[i, 2] * mark_map[j, 2] + mark_map[i, 3] * mark_map[j, 3] > self.config['direct_threshold']:
                    distance = torch.pow(torch.pow(mark_map[i, 0] - mark_map[j, 0], 2) +
                                         torch.pow(mark_map[i, 1] - mark_map[j, 1], 2), 0.5)
                    distance_map[i, j] = distance
                    distance_map[j, i] = distance

        # Slot check
        slot_list = []
        for i in range(0, mark_count - 1):
            for j in range(i + 1, mark_count):
                distance = distance_map[i, j]
                if (distance_map[i] + distance_map[j] < self.config['elliptic_coefficient'] * distance).sum() == 2 \
                        and distance > self.config['distance_threshold']:
                    slot_length = 120 if distance < 80 else 60
                    vx = torch.abs(mark_map[i, 0] - mark_map[j, 0]) / distance
                    vy = torch.abs(mark_map[i, 1] - mark_map[j, 1]) / distance
                    delta_x = -slot_length * vx if mark_map[i, 2] < 0 else slot_length * vx
                    delta_y = -slot_length * vy if mark_map[i, 3] < 0 else slot_length * vy

                    slot_list.append(((mark_map[i, 1], mark_map[i, 0]),
                                      (mark_map[j, 1], mark_map[j, 0]),
                                      (mark_map[j, 1] + delta_x, mark_map[j, 0] + delta_y),
                                      (mark_map[i, 1] + delta_x, mark_map[i, 0] + delta_y)))
                    break
        return slot_list
