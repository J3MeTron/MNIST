"""
MNIST вручную на тензорах PyTorch.

Пакет собран так, чтобы одни и те же функции использовались
и в ноутбуках (включая Google Colab), и при запуске из PyCharm.

Модули:
    data       - загрузка MNIST и нарезка на мини-батчи
    functional - softmax, log_softmax, NLL, ReLU, accuracy (прямой проход)
    grads      - аналитические формулы градиентов и сверка их с autograd
    models     - линейная модель 784->10 и MLP 784->128->10
    train      - цикл обучения и CLI
"""

__all__ = ["data", "functional", "grads", "models", "train"]
