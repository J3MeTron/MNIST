# MNIST вручную на тензорах PyTorch

Учебный проект: классификация рукописных цифр MNIST, собранная из базовых
тензорных операций - без `torch.nn` и без `torch.optim`. Все градиенты
выведены на бумаге, выписаны формулами и проверены против autograd и против
конечных разностей.

Курс "Алгоритмы нейронных сетей", магистратура ВГУ.

## Результаты

| Модель | Параметров | Точность на валидации | Ошибок из 10000 |
|---|---|---|---|
| Линейная 784-10 | 7 850 | 92.43% | 757 |
| MLP 784-128-10 (ReLU) | 101 770 | 97.42% | 258 |
| Тот же MLP через `nn.Module` | 101 770 | 97.20% | 280 |
| MLP + momentum 0.9 | 101 770 | 97.66% | 234 |

Условия: SGD, `batch_size=64`, `lr=0.1`, 10 эпох, `seed=0`. Валидация -
10000 примеров, которых модель не видела при обучении.

Ручная реализация и штатная через `nn.Module` дают одинаковую точность в
пределах шума SGD. Это и есть главный вывод проекта: за вызовами библиотеки
нет ничего, кроме разобранных формул.

---

## С чего начать

Ноутбуки читаются по порядку. Каждый **самодостаточен**: не импортирует
ничего из проекта, все нужное определено в его собственных ячейках. Любой
можно открыть в Colab отдельно от остальных.

| № | Ноутбук | О чем |
|---|---|---|
| 01 | [`notebooks/01_tensors_and_autograd.ipynb`](notebooks/01_tensors_and_autograd.ipynb) | тензоры, broadcasting, вычислительный граф, три грабли autograd, численная проверка градиента |
| 02 | [`notebooks/02_linear_softmax_manual.ipynb`](notebooks/02_linear_softmax_manual.ipynb) | линейная модель: softmax, перекрестная энтропия, **вывод градиентов**, обучение, визуализация весов как шаблонов цифр |
| 03 | [`notebooks/03_mlp_manual.ipynb`](notebooks/03_mlp_manual.ipynb) | MLP: ReLU, **цепное правило через нелинейность**, ручной backward |
| 04 | [`notebooks/04_compare_with_nn.ipynb`](notebooks/04_compare_with_nn.ipynb) | та же сеть через `nn.Module` и `optim.SGD`, построчное сопоставление с ручной версией |

Формулы отдельно от кода, с выводами - в [`docs/theory.md`](docs/theory.md).

---

## Запуск в Google Colab

### Вариант 1: загрузить ноутбук (проще всего)

1. Открыть [colab.research.google.com](https://colab.research.google.com)
2. `Файл -> Загрузить блокнот`, выбрать нужный `.ipynb` из `notebooks/`
3. `Среда выполнения -> Выполнить все`

Ставить зависимости не нужно: `torch`, `matplotlib` и `requests` в Colab
уже есть. Данные (16 МБ) ноутбук скачает сам в первой ячейке загрузки.

GPU не требуется - обе модели обучаются на процессоре за пару минут.

### Вариант 2: с модулями проекта

Нужен только если хочется вызывать `mnist_manual` из ноутбука (последняя
ячейка ноутбука 04). Загрузить архив проекта и распаковать:

```python
from google.colab import files
files.upload()          # выбрать mnist_manual.zip
!unzip -q mnist_manual.zip
import sys; sys.path.insert(0, "src")
```

Архив собирается командой:

```bash
zip -r mnist_manual.zip src notebooks docs requirements.txt README.md
```

### Вариант 3: через Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/MNIST/src")
```

---

## Запуск локально (PyCharm)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

В PyCharm: `Settings -> Project -> Python Interpreter -> Add -> Existing`,
указать `.venv/bin/python`.

Ноутбуки открываются прямо в PyCharm (Professional) либо через
`jupyter notebook`.

Модели можно обучать и без ноутбуков, через CLI:

```bash
python run.py                                   # линейная, 10 эпох
python run.py --model mlp                       # MLP 784-128-10
python run.py --model mlp --epochs 3 --lr 0.2
python run.py --help                            # все параметры
```

`run.py` - обертка, которая добавляет `src` в путь поиска модулей. В
PyCharm достаточно нажать Run на этом файле; флаги задаются в
`Run -> Edit Configurations -> Parameters`.

Параметры CLI: `--model {linear,mlp}`, `--epochs`, `--batch-size`, `--lr`,
`--hidden`, `--seed`, `--log-every`, `--check-grads-every`, `--data-dir`.
Флаг `--check-grads-every 0` отключает сверку ручных градиентов
(по умолчанию - каждые 200 шагов).

---

## Структура проекта

```
MNIST/
├── README.md                  этот файл
├── requirements.txt           зависимости
├── run.py                     запуск обучения из PyCharm
│
├── notebooks/                 ГЛАВНЫЙ МАТЕРИАЛ, читать по порядку
│   ├── 01_tensors_and_autograd.ipynb
│   ├── 02_linear_softmax_manual.ipynb
│   ├── 03_mlp_manual.ipynb
│   └── 04_compare_with_nn.ipynb
│
├── src/mnist_manual/          тот же код модулями, для PyCharm и CLI
│   ├── data.py                загрузка MNIST, нарезка на мини-батчи
│   ├── functional.py          softmax, log_softmax, NLL, ReLU, accuracy
│   ├── grads.py               аналитические градиенты + сверка с autograd
│   ├── models.py              LinearModel и MLPModel (без nn.Module)
│   └── train.py               цикл обучения, ручной SGD, CLI
│
├── docs/
│   ├── theory.md              все формулы с выводами, без кода
│   └── dialog/                журнал работы с ассистентом
│       ├── README.md          формат журнала
│       ├── 00-postanovka.md   исходная задача, разбор ноутбука преподавателя
│       ├── 01-utochnenie-resheniy.md   четыре развилки и их основания
│       └── 02-realizaciya.md  результаты, найденные ошибки
│
├── reference/
│   └── teacher_pytorch_mnist.ipynb     исходный ноутбук преподавателя
│
└── data/                      mnist.pkl.gz, качается сам (в .gitignore)
```

---

## Что именно сделано вручную

| Что | Как в проекте | Чем заменяется в обычном коде |
|---|---|---|
| объявление параметров | `torch.randn(...) / sqrt(n_in)`, `requires_grad_()` | `nn.Linear` |
| прямой проход | `x @ W + b`, `z.clamp(min=0)` | `layer(x)`, `F.relu` |
| функция потерь | `nll(log_softmax(z), y)` | `nn.CrossEntropyLoss` |
| градиенты | формулы в `grads.py` + сверка с `backward()` | только `backward()` |
| шаг спуска | `p -= lr * p.grad` под `no_grad()` | `optimizer.step()` |
| обнуление градиентов | `p.grad.zero_()` | `optimizer.zero_grad()` |
| нарезка на батчи | `torch.randperm` + срезы | `DataLoader` |

Autograd используется (это выбранный подход, см.
[`docs/dialog/01-utochnenie-resheniy.md`](docs/dialog/01-utochnenie-resheniy.md),
развилка 1), но каждая его величина продублирована ручной формулой.
Сверка встроена в цикл обучения через `assert` и срабатывает каждые 200
шагов, то есть формулы проверяются на всем протяжении обучения, а не
только в начальной точке.

### Ключевые формулы

| Что | Формула |
|---|---|
| softmax | $p_k = e^{z_k - m} / \sum_j e^{z_j - m}$, где $m = \max_j z_j$ |
| перекрестная энтропия | $L = -\frac{1}{B}\sum_n \log p_{n, y_n}$ |
| градиент по логитам | $\partial L/\partial Z = (\mathrm{softmax}(Z) - Y)/B$ |
| линейный слой | $\partial L/\partial W = X^\top G$, $\ \partial L/\partial b = \mathrm{sum}(G, 0)$, $\ \partial L/\partial X = G W^\top$ |
| ReLU | $\partial L/\partial Z = (\partial L/\partial H) \odot [Z > 0]$ |
| шаг SGD | $p \leftarrow p - \eta\, \partial L/\partial p$ |

Выводы - в [`docs/theory.md`](docs/theory.md).

---

## Данные

`mnist.pkl.gz` из репозитория туториалов PyTorch - тот же файл, что в
ноутбуке преподавателя. Внутри три набора: train (50000), valid (10000),
test (10000, в проекте не используется).

Картинки уже развернуты в векторы длины 784 и нормированы в диапазон
[0, 1]. Нормировка обязательна: при значениях пикселей 0..255 логиты
получились бы порядка сотен, и softmax переполнился бы на первом шаге.

`torchvision.datasets.MNIST` сознательно не используется - он тянет лишнюю
зависимость и прячет данные за слоем `Dataset`/`DataLoader`, а для разбора
нужен сырой тензор `(N, 784)`.

---

## Журнал работы с ассистентом

Папка [`docs/dialog/`](docs/dialog/) - этапы работы: что запрашивалось,
какие решения приняты и на каком основании, что не сработало с первого
раза. Формат и правила ведения описаны в
[`docs/dialog/README.md`](docs/dialog/README.md).

Самое полезное там - раздел "Проблемы и исправления" в
[`02-realizaciya.md`](docs/dialog/02-realizaciya.md). Например, первая
версия численной проверки градиента давала расхождение 16% вместо
ожидаемых 0.1% по двум независимым причинам, одна из которых (пиксели с
ровно нулевым градиентом, на которых проверка проходит при любой формуле)
в итоге попала в учебный материал как отдельный разбор.
