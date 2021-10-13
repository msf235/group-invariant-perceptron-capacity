import torch
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms


class FakeData(VisionDataset):
    """A fake dataset that returns randomly generated images and returns them
    as pytorch tensors. Based off of torchvision's FakeData, but this version
    does not convert the random tensor to a PIL image.

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the dataset. Default: 10
        transform (callable, optional): A function/transform that  takes in a
            tensor and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(
            self,
            size: int = 1000,
            image_size: Tuple[int, int, int] = (3, 224, 224),
            num_classes: int = 10,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            random_offset: int = 0,
    ) -> None:
        super(FakeData, self).__init__(None, transform=transform,  # type: ignore[arg-type]
                                       target_transform=target_transform)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        # img = 2*torch.rand(*self.image_size)-1
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        # img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target.item()

    def __len__(self) -> int:
        return self.size

def get_shifted_img(img: torch.Tensor, gx: int, gy: int):
    """Receives input image img and returns a shifted version,
    where the shift in the x direction is given by gx and in the y direction
    by gy."""
    img_ret = img.clone()
    img_ret = torch.roll(img_ret, (gx, gy), dims=(-2, -1))
    return img_ret

class SubsampledData(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_idx):
        super().__init__()
        self.dataset = dataset
        self.sample_idx = sample_idx

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, idx):
        item = self.dataset[self.sample_idx[idx]]
        return item

class ShiftDataset2D(torch.utils.data.Dataset):
    """Takes in a normal dataset of images and produces a dataset that
    samples from 2d shifts of this dataset, keeping the label for each
    shifted version of an image the same."""
    def __init__(self, core_dataset, shift_x=1, shift_y=1):
        """
        Parameters
        ----------
        core_dataset : torch.utils.data.Dataset
            The image dataset that we are computing shifts of.
        shift_x : int
            The number of pixels by which the image is shifted in the x
            direction.
        shift_y : int
            The number of pixels by which the image is shifted in the y
            direction.
        """
        super().__init__()
        self.core_dataset = core_dataset
        self.sx, self.sy = self.core_dataset[0][0].shape[1:]
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.sx = self.sx // self.shift_x
        self.sy = self.sy // self.shift_y
        # self.targets = torch.tile(torch.tensor(self.core_dataset.targets),
                                  # (self.sx*self.sy,))  # Too large for memory
        # self.G = itertools.product((range(self.xs), range(sy))) # Lazy approach

    def __len__(self):
        return len(self.core_dataset)*self.sx*self.sy

    def __getitem__(self, idx):
        g_idx = idx % (self.sx*self.sy)
        idx_core = idx // (self.sx*self.sy)
        # g = self.G[g_idx] # Lazy approach
        gx = self.shift_x * (g_idx // self.sy)
        gy = self.shift_y * (g_idx % self.sy)
        img, label = self.core_dataset[idx_core]
        return get_shifted_img(img, gx, gy), label, idx_core

class ShiftDataset1D(torch.utils.data.Dataset):
    """Takes in a normal dataset of images and produces a dataset that
    samples from 1d shifts of this dataset, keeping the label for each
    shifted version of an image the same."""
    def __init__(self, core_dataset, shift_y=1):
        super().__init__()
        self.core_dataset = core_dataset
        self.shift_y = shift_y
        self.sy = self.core_dataset[0][0].shape[-1]
        self.sy = self.sy // self.shift_y
        # self.targets = torch.tile(torch.tensor(self.core_dataset.targets),
                                  # (self.sy,))

    def __len__(self):
        return len(self.core_dataset)*self.sy

    def __getitem__(self, idx):
        g_idx = idx % self.sy
        idx_core = idx // self.sy
        gy = self.shift_y * (g_idx % self.sy)
        img, label = self.core_dataset[idx_core]
        return get_shifted_img(img, 0, gy), label, idx_core
