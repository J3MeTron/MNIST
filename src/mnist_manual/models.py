"""
Две модели, собранные из тензоров без torch.nn.

LinearModel - этап 1: 784 -> 10, softmax-регрессия. Ровно то, что в
    ноутбуке преподавателя, но параметры и прямой проход завернуты
    в объект, чтобы цикл обучения был общим для обеих моделей.

MLPModel - этап 2: 784 -> 128 -> 10 с ReLU между слоями. Добавляется
    скрытый слой, и обратный проход впервые проходит через нелинейность.

Обе модели намеренно НЕ наследуются от torch.nn.Module: параметры -
обычные тензоры с requires_grad=True, список параметров модель отдает
сама. Шаг градиентного спуска тоже написан вручную, torch.optim не
используется.
"""

from __future__ import annotations

import torch

from . import functional as F
from . import grads


class LinearModel:
    """
    Softmax-регрессия: z = x @ W + b, предсказание - argmax по z.

    Параметров: 784 * 10 + 10 = 7850.
    Ожидаемая точность на валидации: около 92 процентов.

    Почему такая модель вообще работает на MNIST. Каждый столбец W -
    это шаблон одной цифры, разложенный в вектор длины 784. Логит
    z_k = сумма произведений яркости пикселей на вес шаблона k, то есть
    мера похожести картинки на шаблон. Для цифр, различимых по "среднему
    силуэту", этого хватает; путаница остается там, где силуэты похожи
    (4 и 9, 3 и 5, 7 и 9).
    """

    name = "linear"

    def __init__(
        self,
        n_in: int = 784,
        n_out: int = 10,
        generator: torch.Generator | None = None,
    ) -> None:
        self.w, self.b = F.init_linear(n_in, n_out, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Логиты z формы (B, C). Softmax тут не применяется намеренно.

        Для предсказания softmax не нужен (argmax от него тот же), а для
        ошибки его применяет cross_entropy. Лишнее применение softmax
        перед cross_entropy - типовая ошибка: получится softmax от
        softmax, и модель будет учиться заметно хуже.
        """
        return x @ self.w + self.b

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def parameters(self) -> dict[str, torch.Tensor]:
        """Обучаемые параметры под теми же именами, что в grads.*."""
        return {"w": self.w, "b": self.b}

    def manual_grads(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Градиенты по аналитическим формулам, без autograd."""
        return grads.backward_linear_model(x, y, self.w, self.b)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters().values())


class MLPModel:
    """
    Один скрытый слой: x -> W1 -> ReLU -> W2 -> логиты.

    Параметров при H = 128: 784*128 + 128 + 128*10 + 10 = 101 770.
    Ожидаемая точность на валидации: около 97-98 процентов.

    Что дает скрытый слой. Линейная модель умеет рисовать между классами
    только плоскости (гиперплоскости в 784-мерном пространстве). Два
    линейных слоя подряд без нелинейности между ними эквивалентны одному
    (произведение двух матриц - снова матрица), поэтому ReLU здесь не
    украшение, а то, что вообще делает сеть нелинейной. Скрытый слой
    строит 128 признаков-детекторов (штрихи, изгибы, замкнутые области),
    а второй слой уже линейно комбинирует их.
    """

    name = "mlp"

    def __init__(
        self,
        n_in: int = 784,
        n_hidden: int = 128,
        n_out: int = 10,
        generator: torch.Generator | None = None,
    ) -> None:
        self.w1, self.b1 = F.init_linear(n_in, n_hidden, generator=generator)
        self.w2, self.b2 = F.init_linear(n_hidden, n_out, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Логиты (B, C). Промежуточные значения не сохраняем: autograd
        держит их сам, а ручной проход в grads.backward_mlp пересчитывает
        их заново - так явно видно, что именно нужно для градиента."""
        z1 = x @ self.w1 + self.b1
        h1 = F.relu(z1)
        return h1 @ self.w2 + self.b2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def parameters(self) -> dict[str, torch.Tensor]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def manual_grads(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        return grads.backward_mlp(x, y, self.w1, self.b1, self.w2, self.b2)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters().values())


def build_model(kind: str = "linear", **kwargs) -> LinearModel | MLPModel:
    """Фабрика по имени - нужна для CLI в train.py."""
    if kind == "linear":
        return LinearModel(**kwargs)
    if kind == "mlp":
        return MLPModel(**kwargs)
    raise ValueError(f"неизвестная модель: {kind!r}, ожидается 'linear' или 'mlp'")
