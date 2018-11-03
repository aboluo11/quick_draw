from lightai.core import *
import torch.multiprocessing as mp
from torch.utils.data.dataloader import pin_memory_batch

def to_cuda_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, container_abcs.Mapping):
        return {k: to_cuda_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, container_abcs.Sequence):
        return [to_cuda_batch(sample) for sample in batch]
    else:
        return batch

def worker(bs, tsfm, df_queue, out_queue):
    while True:
        df = df_queue.get()
        batch = collate([tsfm(df.iloc[i]) for i in range(len(df))], cuda=False)
        batch = pin_memory_batch(batch)
        out_queue.put(batch)

class _Dataset:
    def __init__(self, dataset):
        self.fold_path = dataset.fold_path
        self.cv = dataset.cv
        self.batch_size = dataset.batch_size
        self.num_workers = dataset.num_workers
        self.tsfm = dataset.tsfm
        self.out_queue = mp.Queue()
        self.workers = []
        self.df_queues = []
        self.batches_outstanding = 0
        for _ in range(self.num_workers):
            df_queue = mp.Queue()
            self.df_queues.append(df_queue)
            w = mp.Process(target=worker, args=(self.batch_size, self.tsfm, df_queue, self.out_queue))
            w.daemon = True
            w.start()
            self.workers.append(w)
        self.put_df_gen = self.put_df()
        for _ in range(self.num_workers):
            next(self.put_df_gen)

    def __next__(self):
        if self.batches_outstanding == 0:
            raise StopIteration
        batch = self.out_queue.get()
        batch = to_cuda_batch(batch)
        self.batches_outstanding -= 1
        next(self.put_df_gen)
        return batch

    def __iter__(self):
        return self

    def put_df(self):
        worker_idx = 0
        for k in np.random.permutation(self.cv):
            for df in pd.read_csv(self.fold_path / f'{k}.csv', chunksize=self.batch_size*self.num_workers):
                sampler = BatchSampler(RandomSampler(df), batch_size=self.batch_size, drop_last=False)
                for batch_idx in sampler:
                    self.df_queues[worker_idx].put(df.iloc[batch_idx])
                    worker_idx = (worker_idx + 1) % self.num_workers
                    self.batches_outstanding += 1
                    yield
        yield

class Dataset:
    def __init__(self, fold_path, fold, tsfm, train, batch_size, num_workers):
        self.fold_path = Path(fold_path)
        if train:
            self.cv = list(range(0, fold * 10)) + list(range((fold + 1) * 10, 70))
        else:
            self.cv = list(range(fold * 10, (fold + 1) * 10))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tsfm = tsfm

    def __iter__(self):
        return _Dataset(self)
