import torch
import numpy as np

def test_tile_repeat():
    a = np.random.uniform(0.0, 1.0, size=[5, 3, 4])
    a_t = torch.from_numpy(a)
    a_tiled = np.tile(a, [3, 2, 5])
    a_t_tiled = a_t.repeat(3, 2, 5)
    print(np.linalg.norm(a_tiled - a_t_tiled.numpy()))

    a = np.random.uniform(0.0, 1.0, size=[2, 2, 1])
    a_t = torch.from_numpy(a)
    a_tiled = np.tile(a, [1, 1, 2])
    a_t_tiled = a_t.repeat(1, 1, 2)
    print(a_t_tiled.shape)
    print(np.linalg.norm(a_tiled - a_t_tiled.numpy()))

    b = np.random.uniform(0.0, 1.0, size=[5])
    b_t = torch.from_numpy(b)
    b_tiled = np.tile(b, [2])
    b_t_tiled = b_t.repeat(2)
    print(np.linalg.norm(b_tiled - b_t_tiled.numpy()))

test_tile_repeat()