import torch.multiprocessing as mp
import time

def f():
    print(111)
    yield

f()
