import os
import sys
from pathlib import Path

path = Path(os.getcwd())
sys.path.insert(0, str(path))
path = path.parent.absolute()
sys.path.insert(0, str(path))

import torch
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from ppo.utils import make_env


if __name__ == '__main__':
    
    filepath = os.getcwd() + '/testing/buffer_test'

    load_storage = LazyMemmapStorage(1000, device='cpu')
    rb_load = TensorDictReplayBuffer(storage=load_storage)
    rb_load.loads(filepath)

    print(len(rb_load))

    print(rb_load.sample(5))


