import os
import random
import shutil

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS


class ImagesDataset(Dataset):
    def __init__(self, root, transform=None, loader=default_loader, extensions=IMG_EXTENSIONS):
        super().__init__()
        self.root = root
        self.transform = transform
        self.loader = loader
        self.extensions = extensions
        self._img_paths = [
            entry.path for entry in os.scandir(root) if entry.is_file() and entry.name.lower().endswith(self.extensions)
        ]

    def __getitem__(self, key):
        path = self._img_paths[key]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self._img_paths)


def split_folder(path, train_path, test_path, test_num):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    files = [entry for entry in os.scandir(path) if entry.is_file()]
    random.shuffle(files)
    for i, file in enumerate(files):
        target_dir = test_path if i < test_num else train_path
        new_path = os.path.join(target_dir, file.name)
        shutil.move(file.path, new_path)


if __name__ == "__main__":
    import random
    SEED = 0
    random.seed(SEED)

    # CURRENT_DATASET = 'cucumber'
    CURRENT_DATASET = 'banana'
    TEST_NUM = 50
    os.chdir(os.path.dirname(__file__))
    path = f"./datasets/{CURRENT_DATASET}"
    train_path = f"{path}/train"
    test_path = f"{path}/test"
    split_folder(path, train_path, test_path, TEST_NUM)
