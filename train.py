from src.tsfm import *
from src.data import *
from src.metrics import *
from lightai.core import *


def tsfm(row):
    return np.array(row['y'])

if __name__ == '__main__':
    val_ds = Dataset('inputs/k_fold/', fold=0, tsfm=tsfm, train=False, batch_size=4096, num_workers=6)
    metrics = MAP()
    start = time.time()
    for batch in val_ds:
        predict = torch.rand((batch.shape[0], 340), dtype=torch.float32, device='cuda')
        metrics(predict, batch)
    print(time.time() - start)
    print(metrics.res().item())