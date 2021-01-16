def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = True


class ModelWeightsFreezer:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        freeze_all_params(self.model)
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        unfreeze_all_params(self.model)
        return self.model
