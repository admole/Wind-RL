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


def generate_replay_buffer(
        total_buffer_size,
        rollout_steps,
        policy=None,
        device='cpu',
        dummy_update=False,
):
    env = make_env(device=device, dummy_update=dummy_update)
    storage = LazyMemmapStorage(total_buffer_size, device=device)
    replay_buffer = TensorDictReplayBuffer(storage=storage)
    for _ in range(total_buffer_size // rollout_steps):
        # We perform as many rollouts as are needed to fill the buffer
        td = env.rollout(max_steps=rollout_steps, policy=policy)
        replay_buffer.extend(td)
    return replay_buffer


if __name__ == '__main__':

    rb = generate_replay_buffer(
        total_buffer_size=10000,
        rollout_steps=5000,
        policy=None,
    )

    filepath = os.getcwd() + '/offline/testing/buffer_test'
    rb.dumps(filepath)

    load_storage = LazyMemmapStorage(10000, device='cpu')
    rb_load = TensorDictReplayBuffer(storage=load_storage)
    rb_load.loads(filepath)

    print(len(rb_load))
