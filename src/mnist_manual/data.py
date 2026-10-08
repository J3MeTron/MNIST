"""
Загрузка MNIST и нарезка на мини-батчи.

Берем тот же архив, что в ноутбуке преподавателя: mnist.pkl.gz из
репозитория туториалов PyTorch. Внутри лежит готовый pickle с тремя
наборами: train (50000), valid (10000), test (10000).

Почему не torchvision.datasets.MNIST: тот путь тянет дополнительную
зависимость и скрывает данные за слоем Dataset/DataLoader. Нам нужно
видеть сырой тензор формы (N, 784) - на нем вся арифметика очевидна.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Iterator

import torch

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"


def default_data_dir() -> Path:
    """Папка data в корне проекта (на один уровень выше src)."""
    return Path(__file__).resolve().parents[2] / "data" / "mnist"


def download(data_dir: Path | None = None) -> Path:
    """Скачать mnist.pkl.gz, если его еще нет. Возвращает путь к архиву."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / FILENAME

    if not archive.exists():
        import requests  # импорт внутри: нужен только при первом скачивании

        content = requests.get(URL + FILENAME, timeout=60).content
        archive.write_bytes(content)

    return archive


def load_mnist(
    data_dir: Path | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Загрузить MNIST как четыре тензора.

    Возвращает (x_train, y_train, x_valid, y_valid), где
        x_* : float32, форма (N, 784), значения уже в диапазоне [0, 1]
        y_* : int64,   форма (N,),     метки классов 0..9

    Картинка 28x28 в архиве уже развернута в вектор длины 784 (28 * 28).
    Метки должны быть именно int64 (long): PyTorch требует long для
    индексации и для функций потерь по классам.
    """
    archive = download(data_dir)

    with gzip.open(archive.as_posix(), "rb") as f:
        (x_train, y_train), (x_valid, y_valid), _ = pickle.load(f, encoding="latin-1")

    return (
        torch.tensor(x_train, dtype=dtype),
        torch.tensor(y_train, dtype=torch.int64),
        torch.tensor(x_valid, dtype=dtype),
        torch.tensor(y_valid, dtype=torch.int64),
    )


def iterate_minibatches(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Пройти по данным мини-батчами.

    Замена torch.utils.data.DataLoader на десять строк: вся выборка уже
    в памяти одним тензором, поэтому батч - это просто срез по индексам.

    shuffle=True перемешивает порядок примеров на каждой эпохе. Это важно:
    на фиксированном порядке SGD видит одну и ту же последовательность
    градиентов и сходится хуже.

    Последний батч может быть короче batch_size, если N не делится нацело.
    Мы его не выбрасываем - делим ошибку на фактический размер батча.
    """
    n = x.shape[0]

    if shuffle:
        order = torch.randperm(n, generator=generator)
    else:
        order = torch.arange(n)

    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def describe(x: torch.Tensor, y: torch.Tensor, name: str = "set") -> str:
    """Одна строка со сводкой по набору - удобно печатать в ноутбуке."""
    return (
        f"{name}: x={tuple(x.shape)} {x.dtype} "
        f"[{x.min():.3f}; {x.max():.3f}], "
        f"y={tuple(y.shape)} {y.dtype} "
        f"классы {int(y.min())}..{int(y.max())}"
    )
