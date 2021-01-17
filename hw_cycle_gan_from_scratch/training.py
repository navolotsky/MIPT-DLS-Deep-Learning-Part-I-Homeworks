import os
import uuid
from itertools import chain

import torch
from tqdm import tqdm

from .utils import WeightsFreezer as freezed


def get_last_created_file(dir_path, file_ext) -> os.DirEntry:
    files = [entry for entry in os.scandir(
        dir_path) if entry.is_file() and entry.name.lower().endswith(file_ext)]
    files.sort(key=lambda x: x.stat().st_ctime_ns)
    return files[-1] if files else None


def delete_files_except_last_created(dir_path, file_ext, files_num_to_retain, stride='auto'):
    files = [entry for entry in os.scandir(
        dir_path) if entry.is_file() and entry.name.lower().endswith(file_ext)]
    files.sort(key=lambda x: x.stat().st_ctime_ns)
    files_len_ = len(files)
    if stride == 'auto':
        stride = max(files_len_ // files_num_to_retain, 1)
    inds_to_retain = []
    for stride_num in range(files_num_to_retain):
        i = -1 - stride_num * stride
        if i < -files_len_:
            i = -files_len_
        inds_to_retain.append(i + files_len_)
    for i, file in enumerate(files):
        if i not in inds_to_retain:
            os.remove(file.path)


class CheckpointDirectoryEmpty(ValueError):
    pass


class Trainer:
    _predefined_attrs = (
        'train_checkpoints_dir',
        'checkpoint_name_pattern',
        'model_final_chekpoint_name_template',
        'retain_only_num_checkpoints',
        'checkpoints_saving_stride',
        '_state_attr_names',
        '_training_state'
    )

    def __init__(
            self,
            train_checkpoints_dir='./model_training_checkpoints',
            checkpoint_name_template="checkpoint_{uuid}",
            model_final_chekpoint_name_template="trained_model_final_state",
            retain_only_num_checkpoints=None,
            checkpoints_saving_stride='auto',
            **kwargs):
        self.train_checkpoints_dir = train_checkpoints_dir
        self.checkpoint_name_pattern = checkpoint_name_template
        self.model_final_chekpoint_name_template = model_final_chekpoint_name_template
        self.retain_only_num_checkpoints = retain_only_num_checkpoints
        self.checkpoints_saving_stride = checkpoints_saving_stride
        self._state_attr_names = set(kwargs.keys())
        self._training_state = kwargs

    def __getattr__(self, name):
        if name in self._state_attr_names:
            return self._training_state[name]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name in self._predefined_attrs:
            return super().__setattr__(name, value)
        else:
            self._state_attr_names.add(name)
            self._training_state[name] = value

    def save_state(self):
        state_dict = {}
        for key, val in self._training_state.items():
            typ = type(val)
            # second condition is a kludge for saving criterion with parameters
            if hasattr(val, 'state_dict') and val.state_dict():
                val = val.state_dict()
            state_dict[key] = (typ, val)
        if self.checkpoint_name_pattern == "checkpoint_{uuid}":
            checkpoint_name = self.checkpoint_name_pattern.format(
                uuid=uuid.uuid4().hex)
        else:
            checkpoint_name = self.checkpoint_name_pattern.format(
                **self._training_state)
        checkpoint_name += ".tar"
        os.makedirs(self.train_checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.train_checkpoints_dir, checkpoint_name)
        torch.save(state_dict, checkpoint_path)

    def save_model(self, *, model_attr_name='model'):
        checkpoint_name = self.model_final_chekpoint_name_template
        if self.model_final_chekpoint_name_template != "trained_model_final_state":
            checkpoint_name = self.model_final_chekpoint_name_template.format(
                **self._training_state)
        checkpoint_name += ".pt"
        os.makedirs(self.train_checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.train_checkpoints_dir, checkpoint_name)
        torch.save(getattr(self, model_attr_name).state_dict(), checkpoint_path)

    def load_state(self, path=None):
        self._training_state.clear()
        if path is None:
            try:
                last_checkpoint = get_last_created_file(
                    self.train_checkpoints_dir, '.tar')
            except FileNotFoundError:
                os.makedirs(self.train_checkpoints_dir, exist_ok=True)
                last_checkpoint = None
            if last_checkpoint is None:
                raise CheckpointDirectoryEmpty(
                    "checkpoint directory is empty: ", self.train_checkpoints_dir)
            path = last_checkpoint.path
        state_dict = torch.load(path)
        self._convert_state_dict_to_internal_state(state_dict)
        self._state_attr_names = set(self._training_state.keys())

    def _convert_state_dict_to_internal_state(self, state_dict):
        special_cases = ('model', 'gen_optimizer', 'dis_optimizer')
        for key, (typ, val) in state_dict.items():
            if key not in special_cases:
                setattr(self, key, val)
        typ, val = state_dict['model']
        self.model = model = typ()
        model.load_state_dict(val)
        model.to(self.device)
        typ, val = state_dict['gen_optimizer']
        self.gen_optimizer = gen_optimizer = typ(chain(
            model.forward_generator.parameters(),
            model.backward_generator.parameters()
        ))
        gen_optimizer.load_state_dict(val)
        typ, val = state_dict['dis_optimizer']
        self.dis_optimizer = dis_optimizer = typ(chain(
            model.forward_discriminator.parameters(),
            model.backward_discriminator.parameters()
        ))
        dis_optimizer.load_state_dict(val)

    def run(self):
        losses = getattr(self, 'loss_by_epochs', None)
        if losses is None:
            losses = []
            setattr(self, 'loss_by_epochs', losses)
        
        model = self.model
        was_training = model.training
        model.train()
        
        generators_parameters = [
            *model.forward_generator.parameters(),
            *model.backward_generator.parameters()
        ]

        discriminators_parameters = [
            *model.forward_discriminator.parameters(),
            *model.backward_discriminator.parameters()
        ]
        
        X_len, Y_len = len(self.X_loader), len(self.Y_loader)
        batches_num = max(X_len, Y_len)
        def get_X_iter(): return iter(self.X_loader)
        def get_Y_iter(): return iter(self.Y_loader)
        if X_len > Y_len:
            # pylint: disable=function-redefined
            def get_Y_iter(): return chain(
                *[iter(self.Y_loader) for _ in range(X_len // Y_len + 1)])
        elif X_len < Y_len:
            # pylint: disable=function-redefined
            def get_X_iter(): return chain(
                *[iter(self.X_loader) for _ in range(Y_len // X_len + 1)])

        epoch = getattr(self, 'epoch', -1) + 1
        postfix_kwargs = {'loss': losses[-1] if losses else None}
        with tqdm(range(epoch, self.epochs), initial=epoch, total=self.epochs, desc="epochs", postfix={'loss': losses[-1] if losses else None}) as pbar:
            for self.epoch in pbar:
                epoch_cum_loss = 0
                for j, (X_batch, Y_batch) in tqdm(enumerate(zip(get_X_iter(), get_Y_iter())), total=batches_num, desc="batches"):
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    self.gen_optimizer.zero_grad()
                    model_output = model(X_batch, Y_batch)
                    (
                        fgen_loss, fdis_loss,
                        bgen_loss, bdis_loss,
                        cyc_loss
                    ) = self.criterion(X_batch, Y_batch, *model_output)
                    gens_loss = fgen_loss + bgen_loss + cyc_loss
                    diss_loss = fdis_loss + bdis_loss
                    with freezed(discriminators_parameters):
                        # retain_graph=True is required because
                        # a computation graph node is shared between gen and dis:
                        gens_loss.backward(retain_graph=True)
                    with freezed(generators_parameters):
                        diss_loss.backward()
                    loss = gens_loss + diss_loss
                    epoch_cum_loss += loss.item()
                    self.gen_optimizer.step()
                    if (j + 1) % self.batches_per_discriminators_update == 0:
                        self.dis_optimizer.step()
                        self.dis_optimizer.zero_grad()
                if batches_num % self.batches_per_discriminators_update != 0:
                    self.dis_optimizer.step()
                    self.dis_optimizer.zero_grad()
                postfix_kwargs['loss'] = epoch_mean_loss = epoch_cum_loss / batches_num
                pbar.set_postfix(postfix_kwargs)
                losses.append(epoch_mean_loss)
                self.save_state()
                if self.retain_only_num_checkpoints is not None:
                    delete_files_except_last_created(
                        self.train_checkpoints_dir,
                        '.tar',
                        self.retain_only_num_checkpoints,
                        self.checkpoints_saving_stride
                    )
        self.save_model(model_attr_name='model')
        model.train(was_training)
