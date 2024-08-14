from torch_geometric.data import Data, InMemoryDataset


class MyInMemoryDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super().__init__('./MyDataset', transform)
        self.data, self.slices = self.collate(data_list)
