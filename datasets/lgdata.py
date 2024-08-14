"""
Dataset folder structure:
./TargetTVTBalanced/
├── mean.npy
├── std.npy
├── test
│   ├── 0
│   ├── 1
│   ├── 2
│   └── 3
├── train
│   ├── 0
│   ├── 1
│   ├── 2
│   └── 3
└── val
    ├── 0
    ├── 1
    ├── 2
    └── 3
"""
import os
import json
# main.py 에서 사용됨
from datasets.base import BaseDataset, DatasetSplit


class LGDataset(BaseDataset):

    def _get_image_data(self, subtest: bool = False):
        data_to_iterate = []
        for dataset_name in self.classnames_to_use:
            """
            self.split == 'train' or 'test'
            """
            with open(self.root, 'r') as f:
                json_data = json.load(f)

            data_path = json_data['dataset_list'][dataset_name]
            class_categories = json_data['class_categories'][dataset_name]

            split_path = os.path.join(data_path, self.split.value)

            def make_data_to_iterate(split_path, class_, anomaly: bool):
                data_to_iterate = []
                class_path = os.path.join(split_path, str(class_))
                imgpaths = sorted(os.listdir(class_path))

                anomaly_text = "defect" if anomaly else "good"
                for imgpath in imgpaths:
                    data_to_iterate.append(
                        [dataset_name, anomaly_text, os.path.join(class_path, imgpath), None])

                if subtest:
                    data_to_iterate = data_to_iterate[:5]

                return data_to_iterate

            if self.split.value == "train":
                for c in class_categories['OK']:
                    data_to_iterate.extend(
                        make_data_to_iterate(split_path, c, False))
            elif self.split.value == "test" or self.split.value == "val":
                try:
                    for c in class_categories['OK']:
                        data_to_iterate.extend(
                            make_data_to_iterate(split_path, c, False))
                    for c in class_categories['NG']:
                        data_to_iterate.extend(
                            make_data_to_iterate(split_path, c, True))
                except:
                    pass
            else:
                raise NotImplementedError

        return None, data_to_iterate
