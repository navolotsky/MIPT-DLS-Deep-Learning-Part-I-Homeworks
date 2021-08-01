import os
from itertools import chain

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImagesDataset
from .losses import CycleGANLoss
from .models import CycleGAN
from .reproducibility import enable_reproducibility
from .training import (CheckpointDirectoryEmpty,
                       Trainer)
from .utils import LRLinearlyDecayToZeroFactorFunc
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    import logging
    import sys

    logger = logging.getLogger("training")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(
        "./training_log.txt", mode='a', encoding='utf-8')
    # file_handler = logging.FileHandler(
    #     "./DEV_training_log.txt", mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    CHEKPOINTS_DIR = "./cycle_gan_training_checkpoints"
    CHECKPOINT_NAME_TEMPLATE = "checkpoint_epoch_{epoch}"
    # CHEKPOINTS_DIR = "./cycle_gan_training_checkpoints2"
    # CHECKPOINT_NAME_TEMPLATE = "checkpoint_epoch_current"
    MODEL_FINAL_CHEKPOINT_NAME_TEMPLATE = "model_final_chekpoint"
    RETAIN_ONLY_NUM_CHECKPOINTS = 5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_SMALLER_EDGE_SIZE = 256
    BATCH_SIZE = 1
    # BATCH_SIZE = 2
    NUM_WORKERS = 0
    # NUM_WORKERS = 4
    STAGE_1_EPOCHS = 100
    STAGE_1_LR = 2e-4
    STAGE_2_EPOCHS = 100
    stage_2_lr_mfactor_func = LRLinearlyDecayToZeroFactorFunc(
        STAGE_1_EPOCHS + STAGE_2_EPOCHS - 1, STAGE_1_EPOCHS - 1)
    VERBOSE = True

    BATCHES_PER_DISCRIMINATORS_UPDATE = 50  # image buffer in the paper
    trainer = Trainer(CHEKPOINTS_DIR, CHECKPOINT_NAME_TEMPLATE,
                      MODEL_FINAL_CHEKPOINT_NAME_TEMPLATE, RETAIN_ONLY_NUM_CHECKPOINTS,
                      gen_lr_lambda=[stage_2_lr_mfactor_func],
                      dis_lr_lambda=[stage_2_lr_mfactor_func])
    try:
        trainer.load_state()
        logger.info("Previous state loading succeeded.")
    except CheckpointDirectoryEmpty:
        logger.info(
            "Previous state loading failed: checkpoint folder is empty. Training from scratch.")
        X_dataset_path = "datasets/banana/train"
        Y_dataset_path = "datasets/cucumber/train"
        # X_dataset_path = "DEV_datasets/banana"
        # Y_dataset_path = "DEV_datasets/cucumber"

        transform = transforms.Compose([
            # transforms.Resize(IMAGE_SMALLER_EDGE_SIZE),
            transforms.Resize(
                (IMAGE_SMALLER_EDGE_SIZE, IMAGE_SMALLER_EDGE_SIZE)),
            # transforms.Normalize(),
            transforms.ToTensor()
        ])

        X_dataset = ImagesDataset(X_dataset_path, transform=transform)
        Y_dataset = ImagesDataset(Y_dataset_path, transform=transform)

        # raise_if_no_deterministic causes:
        # Exception has occurred: RuntimeError
        # reflection_pad2d_backward_cuda does not have a deterministic implementation,
        # but you set 'torch.set_deterministic(True)'. You can turn off determinism just for this operation if that's acceptable for your application.
        # You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
        # ------------
        # cudnn_deterministic=True causes Unable to find a valid cuDNN algorithm to run convolution
        enable_reproducibility(raise_if_no_deterministic=False,
                               cudnn_deterministic=False)

        X_loader = DataLoader(X_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        Y_loader = DataLoader(Y_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

        model = CycleGAN().to(DEVICE)

        criterion = CycleGANLoss()

        generators_optimizer = Adam(
            chain(
                model.forward_generator.parameters(),
                model.backward_generator.parameters()
            ), lr=STAGE_1_LR)
        generators_lr_scheduler = LambdaLR(
            generators_optimizer, lr_lambda=stage_2_lr_mfactor_func, verbose=VERBOSE)
        discriminators_optimizer = Adam(
            chain(
                model.forward_discriminator.parameters(),
                model.backward_discriminator.parameters()
            ), lr=STAGE_1_LR)
        discriminators_lr_scheduler = LambdaLR(
            discriminators_optimizer, lr_lambda=stage_2_lr_mfactor_func, verbose=VERBOSE)

        trainer = Trainer(CHEKPOINTS_DIR, CHECKPOINT_NAME_TEMPLATE,
                          MODEL_FINAL_CHEKPOINT_NAME_TEMPLATE, RETAIN_ONLY_NUM_CHECKPOINTS,
                          model=model, X_loader=X_loader, Y_loader=Y_loader, criterion=criterion,
                          gen_optimizer=generators_optimizer, dis_optimizer=discriminators_optimizer,
                          gen_lr_scheduler=generators_lr_scheduler, dis_lr_scheduler=discriminators_lr_scheduler,
                          epochs=STAGE_1_EPOCHS + STAGE_2_EPOCHS, epochs_num_per_stages=(STAGE_1_EPOCHS, STAGE_2_EPOCHS),
                          batches_per_discriminators_update=BATCHES_PER_DISCRIMINATORS_UPDATE, device=DEVICE)

    runs = 0
    max_tries = 3
    while True:
        runs += 1
        try:
            if runs == 1:
                logger.info("Training started.")
            else:
                logger.info("Training restarted: run try %s/%s.",
                            runs, max_tries)
            trainer.run()
            logger.info("Training seems to be successfully finished.")
            break
        except Exception:
            logger.exception("Some error appeared during training.")
            if runs >= max_tries:
                logger.error(
                    "All %s run tries exceeded. Stopping process.", max_tries)
                sys.exit(1)
            try:
                trainer.load_state()
                logger.info("Training state reloading succeeded.")
            except CheckpointDirectoryEmpty:
                pass
    trainer.save_state()
    logger.info("Final state saving succeeded.")
