"""
Цикл обучения: ручной SGD плюс постоянная сверка градиентов с autograd.

Один шаг обучения состоит из пяти действий:
    1. взять мини-батч (x, y)
    2. прямой проход: z = model(x), ошибка loss = cross_entropy(z, y)
    3. обратный проход: loss.backward() заполняет .grad у параметров
    4. шаг спуска: p -= lr * p.grad
    5. обнулить .grad

Пункт 5 обязателен и часто забывается. PyTorch НЕ затирает .grad, а
прибавляет к нему (это сделано ради сетей, где один параметр
используется несколько раз). Без обнуления в .grad накопится сумма
градиентов по всем прошедшим батчам, шаги станут огромными, и ошибка
уйдет в NaN за несколько десятков итераций.

Пункт 4 обернут в torch.no_grad(). Внутри него PyTorch не записывает
операции в граф. Без него обновление веса само стало бы частью графа:
получился бы параметр, зависящий от предыдущего параметра, граф рос бы
с каждым шагом, память кончилась бы, а градиенты потеряли смысл.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import data as data_mod
from . import functional as F
from . import grads as grads_mod
from .models import LinearModel, MLPModel, build_model


@dataclass
class History:
    """Историю держим в простых списках - их удобно рисовать matplotlib."""

    step: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    valid_acc: list[float] = field(default_factory=list)
    grad_check: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model: LinearModel | MLPModel,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1000,
) -> tuple[float, float]:
    """
    Средняя ошибка и точность на всем наборе.

    Считаем батчами, чтобы не держать матрицу (10000, 128) целиком -
    на MNIST это не критично, но привычка верная.

    Усреднение взвешенное по размеру батча: последний батч может быть
    короче, и простое среднее средних дало бы ему лишний вес.

    Декоратор @torch.no_grad() отключает построение графа на все время
    функции: при оценке градиенты не нужны, а без графа это быстрее
    и не расходует память.
    """
    total_loss = 0.0
    total_correct = 0.0
    total_n = 0

    for xb, yb in data_mod.iterate_minibatches(x, y, batch_size, shuffle=False):
        z = model(xb)
        bs = yb.shape[0]
        total_loss += float(F.cross_entropy(z, yb)) * bs
        total_correct += float(F.accuracy(z, yb)) * bs
        total_n += bs

    return total_loss / total_n, total_correct / total_n


def sgd_step(model: LinearModel | MLPModel, lr: float) -> None:
    """
    Шаг градиентного спуска и обнуление градиентов.

    p -= lr * p.grad - это и есть все обучение. Градиент указывает
    направление наибольшего РОСТА ошибки, поэтому идем против него,
    то есть вычитаем.

    Длину шага задает lr. Слишком маленький - обучение тянется;
    слишком большой - шаг перелетает минимум, ошибка растет или
    улетает в NaN. Для этих моделей рабочий диапазон 0.1 ... 0.5.
    """
    with torch.no_grad():
        for p in model.parameters().values():
            p -= lr * p.grad
            p.grad.zero_()


def train(
    model: LinearModel | MLPModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.1,
    check_grads_every: int = 200,
    log_every: int = 200,
    generator: torch.Generator | None = None,
    verbose: bool = True,
) -> History:
    """
    Обучить модель и вернуть историю.

    check_grads_every - как часто сверять ручные формулы с autograd.
    Проверка стоит примерно один дополнительный прямой проход, поэтому
    делать ее на каждом шаге смысла нет. 0 отключает сверку.

    В History.grad_check попадает максимальное относительное расхождение
    по всем параметрам. Если оно держится на уровне 1e-6 и ниже -
    ручные формулы верны. Если где-то подскочило до 1e-2 - ошибка
    в выводе формул, а не численный шум.
    """
    history = History()
    step = 0

    for epoch in range(1, epochs + 1):
        for xb, yb in data_mod.iterate_minibatches(
            x_train, y_train, batch_size, shuffle=True, generator=generator
        ):
            # 1-2. прямой проход
            z = model(xb)
            loss = F.cross_entropy(z, yb)

            # 3. обратный проход средствами autograd
            loss.backward()

            # сверка: те же градиенты, посчитанные руками
            if check_grads_every and step % check_grads_every == 0:
                manual = model.manual_grads(xb, yb)
                report = grads_mod.compare_with_autograd(manual, model.parameters())
                worst = max(float(r["max_rel_diff"]) for r in report.values())
                history.grad_check.append(worst)

                if not all(r["allclose"] for r in report.values()):
                    raise AssertionError(
                        "ручные градиенты разошлись с autograd на шаге "
                        f"{step}:\n{grads_mod.format_report(report)}"
                    )

            # 4-5. шаг спуска и обнуление .grad
            sgd_step(model, lr)

            history.step.append(step)
            history.train_loss.append(float(loss.detach()))

            if verbose and log_every and step % log_every == 0:
                print(f"  эпоха {epoch}  шаг {step:>5}  ошибка {float(loss.detach()):.4f}")

            step += 1

        v_loss, v_acc = evaluate(model, x_valid, y_valid)
        history.epoch.append(epoch)
        history.valid_loss.append(v_loss)
        history.valid_acc.append(v_acc)

        if verbose:
            print(
                f"эпоха {epoch}/{epochs}: "
                f"валидация ошибка {v_loss:.4f}, точность {v_acc * 100:.2f}%"
            )

    return history


def confusion_matrix(
    model: LinearModel | MLPModel,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Матрица ошибок: в ячейке [i, j] - сколько раз истинный класс i
    был предсказан как j. На диагонали правильные ответы.

    Нужна, чтобы увидеть, какие цифры модель путает между собой,
    а не только итоговый процент.
    """
    with torch.no_grad():
        preds = torch.argmax(model(x), dim=1)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for true_cls, pred_cls in zip(y.tolist(), preds.tolist()):
        cm[true_cls, pred_cls] += 1

    return cm


def main(argv: list[str] | None = None) -> int:
    """CLI для запуска из PyCharm или терминала."""
    parser = argparse.ArgumentParser(
        description="Обучение MNIST на тензорах, без torch.nn и torch.optim"
    )
    parser.add_argument(
        "--model", choices=["linear", "mlp"], default="linear",
        help="linear: 784->10; mlp: 784->128->10 с ReLU",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="шаг обучения; 0.1 устойчив для обеих моделей, "
             "на 0.5 точность начинает колебаться",
    )
    parser.add_argument("--hidden", type=int, default=128, help="только для mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-every", type=int, default=200,
        help="как часто печатать ошибку на батче; 0 отключает",
    )
    parser.add_argument(
        "--check-grads-every", type=int, default=200,
        help="0 отключает сверку ручных градиентов с autograd",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    print("Загрузка MNIST...")
    x_train, y_train, x_valid, y_valid = data_mod.load_mnist(args.data_dir)
    print(" ", data_mod.describe(x_train, y_train, "train"))
    print(" ", data_mod.describe(x_valid, y_valid, "valid"))

    kwargs = {"generator": generator}
    if args.model == "mlp":
        kwargs["n_hidden"] = args.hidden
    model = build_model(args.model, **kwargs)

    print(f"\nМодель: {model.name}, обучаемых параметров: {model.num_params()}")
    base_loss, base_acc = evaluate(model, x_valid, y_valid)
    print(
        f"До обучения: ошибка {base_loss:.4f}, точность {base_acc * 100:.2f}% "
        f"(случайная модель дает около 10%)\n"
    )

    history = train(
        model, x_train, y_train, x_valid, y_valid,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        check_grads_every=args.check_grads_every, log_every=args.log_every,
        generator=generator,
    )

    if history.grad_check:
        print(
            f"\nСверка с autograd: {len(history.grad_check)} проверок, "
            f"худшее относительное расхождение {max(history.grad_check):.3e}"
        )

    print("\nМатрица ошибок на валидации (строка - истина, столбец - предсказание):")
    cm = confusion_matrix(model, x_valid, y_valid)
    print("     " + " ".join(f"{j:>5}" for j in range(10)))
    for i, row in enumerate(cm.tolist()):
        print(f"{i:>3}: " + " ".join(f"{v:>5}" for v in row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
