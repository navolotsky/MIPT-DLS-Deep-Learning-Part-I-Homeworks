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
