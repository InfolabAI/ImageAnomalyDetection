import random
from enum import Enum

import PIL
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RandomRotation90(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        degree = torch.randint(0, 4, (1,)).item()
        ret = torch.rot90(img, degree, dims=[1, 2])
        return ret


class RandomCropMore(torch.nn.Module):
    def __init__(self, img_size, crop_prob=0.5, crop_range=(0.5, 1.0)):
        super().__init__()
        self.crop_prob = crop_prob
        self.crop_range = crop_range
        self.img_size = img_size

    def forward(self, img):
        prob = torch.rand(1).item()
        if prob > self.crop_prob:
            return img
        crop_size_prob = torch.rand(1).item()
        crop_size = crop_size_prob * \
            (self.crop_range[1]-self.crop_range[0]) + self.crop_range[0]
        crop_size = int(crop_size * self.img_size)
        img = transforms.RandomCrop(crop_size)(img)
        return img


class BaseDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        subtest: bool = False,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.root = self.source = source
        self.split = split
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.args = kwargs['args']
        if isinstance(classname, list):
            self.classnames_to_use = classname
        else:
            self.classnames_to_use = [classname]
        self.few_shot_mode = False
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data(
            subtest)

        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(
                brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_img_few = [
            transforms.Resize(resize),
            RandomCropMore(resize),
            transforms.RandomHorizontalFlip(0.25),
            transforms.RandomVerticalFlip(0.25),
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            RandomRotation90(),  # Tensor 이어야 작동함
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img_few = transforms.Compose(self.transform_img_few)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        if self.few_shot_mode:
            image = self.transform_img_few(image)
        else:
            image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self, subtest: bool = False):
        if not self.few_shot_mode:
            _, data_to_iterate = self._get_image_data(subtest)
            return None, data_to_iterate
        else:
            _, data_to_iterate1 = self._get_image_data(subtest)
            self.split = DatasetSplit.TEST
            _, data_to_iterate2 = self._get_image_data(subtest)

            data_to_iterate2_wo_good = []
            for data in data_to_iterate2:
                if data[1] != "good":
                    data_to_iterate2_wo_good.append(data)

            # shuffle data_to_iterate2_wo_good
            random.shuffle(data_to_iterate2_wo_good)

            data_to_iterate = data_to_iterate1 + \
                data_to_iterate2_wo_good[:self.args.n_abnormal]
            print(data_to_iterate)
            return None, data_to_iterate

    @classmethod
    def get_classname(cls):
        return cls._CLASSNAMES
