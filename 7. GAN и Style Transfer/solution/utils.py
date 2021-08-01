from torch.nn import Module


def freeze_all_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_all_params(params):
    for param in params:
        param.requires_grad = True


class WeightsFreezer:
    """
    Context manager which freezes given params.

    Args:
        it: model, models iterable, params iterable
    """

    def __init__(self, it):
        if isinstance(it, Module):
            self.params = list(it.parameters())
        else:
            self.params = []
            for el in iter(it):
                if isinstance(el, Module):
                    self.params.extend(el.parameters())
                else:
                    self.params.append(el)

    def __enter__(self):
        freeze_all_params(self.params)
        return self.params

    def __exit__(self, exc_type, exc_value, traceback):
        unfreeze_all_params(self.params)
        return self.params


class LRLinearlyDecayToZeroFactorFunc:
    def __init__(self, last_epoch_num, first_epoch_num=0):
        self.first_epoch_num = first_epoch_num
        self.last_epoch_num = last_epoch_num

    def __call__(self, current_epoch_num):
        if current_epoch_num < self.first_epoch_num:
            return 1
        return (
            (self.last_epoch_num + 1 - current_epoch_num) /
            (self.last_epoch_num + 1 - self.first_epoch_num))
