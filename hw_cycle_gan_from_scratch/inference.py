import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImagesDataset
from .models import CycleGAN
from . import *


def load_model(checkpoint_path, *, checkpoint_type='state_dict', state_dict_model_key='model', device='cuda'):
    device = torch.device(device)
    if checkpoint_type == 'state_dict':
        model = CycleGAN()
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    elif checkpoint_type == 'entire_model':
        model = torch.load(checkpoint_path)
    elif checkpoint_type == 'general':
        model = CycleGAN()
        state_dict = torch.load(checkpoint_path)
        try:
            state_dict = state_dict[state_dict_model_key][1]
        except KeyError:
            raise ValueError(
                f"no such state_dict_model_key='{state_dict_model_key}' in given state_dict")
        model.load_state_dict(state_dict)
    else:
        raise ValueError(
            'checkpoint_type must be on of ("state_dict", "entire_model", "general")')
    return model.to(device)


def get_data_loader(
        data_dir, dataset_cls=ImagesDataset, transform=transforms.ToTensor(),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    dataset = dataset_cls(data_dir, transform=transform)
    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader


def do_tranformations(model, data_loader, output_dir, direction='forward', device='cuda'):
    device = torch.device(device)
    was_training = model.training
    model.eval()
    to_pil = transforms.ToPILImage()
    image_name_teplate = "image_{image_num:0>" + \
        str(len(str(len(data_loader.dataset)))) + "}"
    image_path_template = os.path.join(output_dir, image_name_teplate)
    if direction == 'forward':
        fgen = model.forward_generator
        bgen = model.backward_generator
    elif direction == 'backward':
        fgen = model.backward_generator
        bgen = model.forward_generator
    else:
        raise ValueError('direction must be "forward" or "backward"')
    with torch.no_grad():
        image_num = -1
        for batch in data_loader:
            batch = batch.to(device)
            transformed = fgen(batch)
            reconstructed = bgen(transformed)
            for o, t, r in zip(batch, transformed, reconstructed):
                image_num += 1
                image_path = image_path_template.format(image_num=image_num)
                to_pil(o).save(image_path + "_0_original.jpg")
                to_pil(t).save(image_path + "_1_transformed.jpg")
                to_pil(r).save(image_path + "_2_reconstructed.jpg")
    model.train(was_training)


def main():
    # model
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "cycle_gan_training_checkpoints",
        "checkpoint_epoch_3.tar",
    )
    model = load_model(
        checkpoint_path=checkpoint_path,
        checkpoint_type='general'
    )

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    # banana -> cucumber
    data_dir = os.path.join(
        os.path.dirname(__file__),
        "datasets/banana/test"
    )
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "inference_result/banana-to-cucumber/test"
    )
    os.makedirs(output_dir, exist_ok=True)
    data_loader = get_data_loader(data_dir, transform=transform)
    do_tranformations(model, data_loader,
                      output_dir,
                      direction='forward')

    # cucumber -> banana
    data_dir = os.path.join(
        os.path.dirname(__file__),
        "datasets/cucumber/test"
    )
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "inference_result/cucumber-to-banana/test"
    )
    os.makedirs(output_dir, exist_ok=True)
    data_loader = get_data_loader(data_dir, transform=transform)
    do_tranformations(model, data_loader,
                      output_dir,
                      direction='backward')


if __name__ == "__main__":
    main()
