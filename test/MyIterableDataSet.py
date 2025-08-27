import math

from torch.utils.data import IterableDataset, DataLoader


class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

from torch.utils.data import get_worker_info

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


ds = MyIterableDataset(start=3, end=7)
print(list(DataLoader(ds, num_workers=0)))
# 系统不支持多进程操作
# print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
# print(list(DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))


