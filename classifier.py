from torch.utils.data import IterableDataset, DataLoader
import random


class StreamDS(IterableDataset):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        # stream records from many files; yield (x, y)
        for f in self.files:
            with open(f, "rb") as fh:
                for rec in read_records(fh):
                    yield decode(rec)


loader = DataLoader(StreamDS(list_of_files), batch_size=64, num_workers=8)
