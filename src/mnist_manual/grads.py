"""
Обратный проход, выписанный руками, и сверка его с autograd.

Это центральный модуль проекта. Градиенты в обучении считает
loss.backward(), но каждая его величина здесь продублирована
аналитической формулой, выведенной на бумаге, и проверена на равенство.

Обозначения (B - батч, D - признаки, H - скрытый слой, C - классы):
    X : (B, D)        вход
    Z : (B, C)        логиты
    P : (B, C)        P = softmax(Z), предсказанные вероятности
    Y : (B, C)        one-hot правильных классов
    G : (B, C)        G = dL/dZ - градиент ошибки по логитам

Главная формула, из которой растет все остальное:

    L = -(1/B) * sum_n log P[n, y_n]
    dL/dZ = (P - Y) / B

Вывод для одного примера. Пусть y - правильный класс, тогда
    L_n = -log P_y = -Z_y + log sum_j exp(Z_j)
Дифференцируем по произвольному Z_k:
    d(-Z_y)/dZ_k            = -[k == y]
    d(log sum_j exp Z_j)/dZ_k = exp(Z_k) / sum_j exp(Z_j) = P_k
Итого dL_n/dZ_k = P_k - [k == y], то есть по всей матрице: P - Y.
Деление на B появляется потому, что в L стоит среднее, а не сумма.

Смысл формулы: градиент по логиту - это "насколько предсказание
превысило правду". Для правильного класса P_y - 1 < 0, логит надо
поднять. Для остальных P_k - 0 > 0, логиты надо опустить.
"""

from __future__ import annotations

import torch

from . import functional as F


# --------------------------------------------------------------------------
# общий кирпич: градиент перекрестной энтропии по логитам
# --------------------------------------------------------------------------

def grad_logits(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    dL/dZ для L = cross_entropy(Z, y). Форма результата совпадает с Z.

    Возвращает (softmax(Z) - one_hot(y)) / B.

    .detach() нужен, если Z пришел из графа autograd: сама формула
    не должна попасть в граф, иначе backward посчитает производные
    еще и от нашей проверки.
    """
    batch_size = y.shape[0]
    num_classes = z.shape[1]

    p = F.softmax(z.detach())
    y_onehot = F.one_hot(y, num_classes)

    return (p - y_onehot) / batch_size


# --------------------------------------------------------------------------
# линейный слой: как градиент проходит через z = x @ W + b
# --------------------------------------------------------------------------

def grad_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Прогнать градиент через линейный слой z = x @ W + b.

    Вход:
        x      : (B, D)  вход слоя
        w      : (D, C)  веса слоя (нужны только для градиента по входу)
        grad_z : (B, C)  уже известный dL/dz на выходе слоя
    Выход:
        grad_w : (D, C)  dL/dW
        grad_b : (C,)    dL/db
        grad_x : (B, D)  dL/dx, передается в предыдущий слой

    Вывод формул. Поэлементно z[n,k] = sum_i x[n,i] * W[i,k] + b[k].

    1) dL/dW[i,k]. Вес W[i,k] участвует в z[n,k] для каждого примера n,
       и dz[n,k]/dW[i,k] = x[n,i]. По правилу суммирования вкладов:
           dL/dW[i,k] = sum_n grad_z[n,k] * x[n,i]
       Это в точности элемент [i,k] произведения x^T @ grad_z,
       поэтому grad_w = x.T @ grad_z, форма (D,B) @ (B,C) = (D,C).

    2) dL/db[k]. Смещение b[k] входит в z[n,k] с коэффициентом 1 для
       каждого n, значит dL/db[k] = sum_n grad_z[n,k]. Это сумма по оси
       батча: grad_z.sum(dim=0), форма (C,).

    3) dL/dx[n,i]. Значение x[n,i] влияет на все логиты строки n:
           dL/dx[n,i] = sum_k grad_z[n,k] * W[i,k]
       что равно элементу [n,i] произведения grad_z @ W^T,
       форма (B,C) @ (C,D) = (B,D).

    Проверка на формах - лучший способ не перепутать порядок множителей:
    из (B,D) и (B,C) матрицу (D,C) можно собрать только как x.T @ grad_z.
    """
    grad_w = x.detach().T @ grad_z
    grad_b = grad_z.sum(dim=0)
    grad_x = grad_z @ w.detach().T

    return grad_w, grad_b, grad_x


# --------------------------------------------------------------------------
# полный обратный проход: модель 784 -> 10
# --------------------------------------------------------------------------

def backward_linear_model(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Все градиенты линейной модели z = x @ W + b одним вызовом.

    Возвращает {"w": dL/dW, "b": dL/db}.

    Цепочка короткая: L <- Z <- (W, b).
        G      = (softmax(Z) - Y) / B
        dL/dW  = X^T @ G
        dL/db  = sum_n G[n]
    """
    z = x.detach() @ w.detach() + b.detach()
    g = grad_logits(z, y)
    grad_w, grad_b, _ = grad_linear(x, w, g)

    return {"w": grad_w, "b": grad_b}


# --------------------------------------------------------------------------
# полный обратный проход: MLP 784 -> H -> 10
# --------------------------------------------------------------------------

def backward_mlp(
    x: torch.Tensor,
    y: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Все градиенты сети x -> линейный -> ReLU -> линейный -> softmax.

    Прямой проход:
        Z1 = X  @ W1 + b1     (B, H)
        H1 = relu(Z1)         (B, H)
        Z2 = H1 @ W2 + b2     (B, C)
        L  = cross_entropy(Z2, y)

    Обратный проход идет строго в обратном порядке. Цепное правило тут
    работает как конвейер: каждый слой получает градиент по своему выходу
    и обязан выдать градиент по своим параметрам и по своему входу.

        G2  = (softmax(Z2) - Y) / B      dL/dZ2    (B, C)
        dW2 = H1^T @ G2                            (H, C)
        db2 = sum_n G2[n]                          (C,)
        dH1 = G2 @ W2^T                  dL/dH1    (B, H)
        G1  = dH1 * (Z1 > 0)             dL/dZ1    (B, H)
        dW1 = X^T @ G1                             (D, H)
        db1 = sum_n G1[n]                          (H,)

    Единственное новое место по сравнению с линейной моделью - переход
    через ReLU. Производная ReLU поэлементная, поэтому цепное правило
    превращается не в матричное умножение, а в поэлементное умножение
    на маску (Z1 > 0): там, где нейрон был закрыт, градиент не проходит
    вообще. Отсюда и эффект "мертвых нейронов": если нейрон ушел в
    отрицательную зону на всех примерах, его веса перестают меняться.

    Возвращает {"w1": ..., "b1": ..., "w2": ..., "b2": ...}.
    """
    xd = x.detach()

    # прямой проход, параметры отцеплены от графа - здесь мы считаем сами
    z1 = xd @ w1.detach() + b1.detach()
    h1 = F.relu(z1)
    z2 = h1 @ w2.detach() + b2.detach()

    # обратный проход
    g2 = grad_logits(z2, y)
    grad_w2, grad_b2, grad_h1 = grad_linear(h1, w2, g2)

    g1 = grad_h1 * F.relu_mask(z1)
    grad_w1, grad_b1, _ = grad_linear(xd, w1, g1)

    return {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}


# --------------------------------------------------------------------------
# сверка ручных формул с autograd
# --------------------------------------------------------------------------

def compare_with_autograd(
    manual: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, dict[str, float | bool]]:
    """
    Сравнить ручные градиенты с теми, что положил в .grad вызов backward().

    manual - результат backward_* из этого модуля,
    params - те же параметры под теми же именами, после loss.backward().

    Для каждого имени считается:
        max_abs_diff - максимальное расхождение по модулю
        max_rel_diff - оно же, отнесенное к масштабу autograd-градиента
        allclose     - проходит ли torch.allclose с заданными допусками

    Почему не требуем точного равенства: порядок операций у autograd
    и у наших формул разный, а сложение float32 неассоциативно
    ((a+b)+c не равно a+(b+c) с точностью до последних битов). Поэтому
    расхождение порядка 1e-7 - норма, а вот 1e-2 означает ошибку в выводе.
    """
    report: dict[str, dict[str, float | bool]] = {}

    for name, manual_grad in manual.items():
        param = params[name]

        if param.grad is None:
            raise ValueError(
                f"у параметра '{name}' .grad пустой: "
                "сначала нужно вызвать loss.backward()"
            )

        auto_grad = param.grad
        diff = (manual_grad - auto_grad).abs()
        scale = auto_grad.abs().max().clamp(min=1e-12)

        report[name] = {
            "max_abs_diff": float(diff.max()),
            "max_rel_diff": float(diff.max() / scale),
            "allclose": bool(
                torch.allclose(manual_grad, auto_grad, rtol=rtol, atol=atol)
            ),
        }

    return report


def format_report(report: dict[str, dict[str, float | bool]]) -> str:
    """Человекочитаемая таблица отчета для печати в ноутбуке."""
    lines = [f"{'параметр':<10} {'max |разн|':>12} {'отн.разн':>12}  вердикт"]
    lines.append("-" * 52)

    for name, row in report.items():
        verdict = "совпало" if row["allclose"] else "РАСХОЖДЕНИЕ"
        lines.append(
            f"{name:<10} {row['max_abs_diff']:>12.3e} "
            f"{row['max_rel_diff']:>12.3e}  {verdict}"
        )

    return "\n".join(lines)
