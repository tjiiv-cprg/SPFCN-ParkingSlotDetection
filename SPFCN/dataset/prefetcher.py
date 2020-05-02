import torch
from torch.utils.data import DataLoader


class DataPrefetcher(object):
    def __init__(self, dataset, batch_size, shuffle, device):
        self.stream = torch.cuda.Stream(device=device)
        self.device = device

        self.loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,
                                 num_workers=4, pin_memory=True)
        self.fetcher = None
        self.next_images = None
        self.next_labels = None

    def refresh(self):
        self.fetcher = iter(self.loader)
        self.preload()

    def preload(self):
        try:
            self.next_images, self.next_labels = next(self.fetcher)
        except StopIteration:
            self.next_images = None
            self.next_labels = None
        else:
            with torch.cuda.stream(self.stream):
                self.next_images = self.next_images.to(self.device).float()
                self.next_labels = self.next_labels.to(self.device).float()

    def next(self):
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        current_images = self.next_images
        current_labels = self.next_labels
        if current_images is not None:
            current_images.record_stream(torch.cuda.current_stream(device=self.device))
            current_labels.record_stream(torch.cuda.current_stream(device=self.device))
        self.preload()
        return current_images, current_labels
