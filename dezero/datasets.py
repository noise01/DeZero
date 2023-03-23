from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable


import numpy as np


class Dataset:
    def __init__(
        self, train=True, transform: Callable = None, target_transform: Callable = None
    ) -> None:
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data: list = None
        self.label: list = None
        self.prepare()

    def __getitem__(self, index: slice) -> np.ndarray:
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(
                self.label[index]
            )

    def __len__(self) -> int:
        return len(self.data)

    def prepare(self) -> None:
        pass


# def get_spiral(train=True) -> tuple[np.ndarray]:
#     seed = 1984 if train else 2020
#     np.random.seed(seed=seed)

#     num_data, num_class, input_dim = 100, 3, 2
#     data_size = num_class * num_data

#     x = np.zeros((data_size, input_dim), dtype=np.float32)
#     t = np.zeros(data_size, dtype=np.int)

#     for j in range(num_class):
#         for i in range(num_data):
#             rate = i / num_data
#             radius = 1.0 * rate
#             theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
#             ix = num_data * j + i
#             x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
#             t[ix] = j

#     indices = np.random.permutation(num_data * num_class)
#     x = x[indices]
#     t = t[indices]
#     return x, t

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self) -> None:
        self.data, self.label = get_spiral
