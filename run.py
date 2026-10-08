#!/usr/bin/env python3
"""
Точка входа для запуска из PyCharm (кнопка Run на этом файле).

Делает только одно: добавляет src в путь поиска модулей и передает
управление в mnist_manual.train. Нужен потому, что иначе пришлось бы
каждый раз выставлять PYTHONPATH=src.

Примеры запуска из терминала:
    python run.py                                  линейная модель, 10 эпох
    python run.py --model mlp                      MLP 784-128-10
    python run.py --model mlp --epochs 3 --lr 0.2
    python run.py --help                           все параметры

Из PyCharm: Run -> Edit Configurations -> Parameters, туда те же флаги.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mnist_manual.train import main  # noqa: E402  (импорт после правки sys.path)

if __name__ == "__main__":
    sys.exit(main())
