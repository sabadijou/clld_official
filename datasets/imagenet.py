'''
Main Source: https://github.com/dev-sungman/Propagate-Yourself-Pytorch
Modified for CLoS
'''

from utils.random_drop import RandomDrop
from torch.utils.data import Dataset
from utils.transforms import *
import numpy as np
import torch
import os


class CLoSDataSet(Dataset):
    def __init__(self, root,
                 args):
        self.root = root
        self.data_size = args.dataset['sample_size']
        self.args = args

        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._make_dataset(self.root, self.class_to_idx)
        self.targets = [s[1] for s in self.samples]

        self.masking = RandomDrop(self.args.dataset['_lambda'],
                                              self.args.dataset['_gamma'],
                                              self.args.dataset['information_loss'],
                                              self.data_size[0],
                                              self.data_size[1])

        self.transform = transforms.Compose([
            transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(prob=0.3, mag=3),
            Solarize(prob=0.3, mag=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, directory, class_to_idx):
        instances = []
        directory = os.path.expanduser(directory)

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

        return instances

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self._load_image(path)
        sample_augmented = self.masking(np.asarray(sample))
        sample_augmented = Image.fromarray(sample_augmented)

        choose_sample = random.choice([True, False])
        if choose_sample:
            sample1, x1, y1, w1, h1 = RandomResizedCrop(self.data_size)(sample)
            sample1, is_flip1 = RandomHorizontalFlip(p=0.5)(sample1)
            sample1 = self.transform(sample1)

            sample2, x2, y2, w2, h2 = RandomResizedCrop(self.data_size)(sample_augmented)
            sample2, is_flip2 = RandomHorizontalFlip(p=0.5)(sample2)
            sample2 = self.transform(sample2)

        else:
            sample1, x1, y1, w1, h1 = RandomResizedCrop(self.data_size)(sample_augmented)
            sample1, is_flip1 = RandomHorizontalFlip(p=0.5)(sample1)
            sample1 = self.transform(sample1)

            sample2, x2, y2, w2, h2 = RandomResizedCrop(self.data_size)(sample)
            sample2, is_flip2 = RandomHorizontalFlip(p=0.5)(sample2)
            sample2 = self.transform(sample2)

        p_base = torch.FloatTensor(np.array([x1, y1, w1, h1]))
        p_moment = torch.FloatTensor(np.array([x2, y2, w2, h2]))
        f_base = torch.FloatTensor(np.array([is_flip1]))
        f_moment = torch.FloatTensor(np.array([is_flip2]))

        base_matrix = self.mat_affine(p_base)
        moment_matrix = self.mat_affine(p_moment)

        if f_base.item() > 0:
            base_matrix = torch.fliplr(base_matrix)
        if f_moment.item() > 0:
            moment_matrix = torch.fliplr(moment_matrix)

        base_A_matrix = self._get_A_matrix(base_matrix, moment_matrix, p_base)
        moment_A_matrix = self._get_A_matrix(moment_matrix, base_matrix, p_moment)

        return (sample1, sample2), (base_A_matrix, moment_A_matrix)

    def mat_affine(self, p, size=7):
        x, y, w, h = p
        matrix = torch.zeros((size, size, 2))
        matrix[:, :, 1] = torch.stack([torch.linspace(x, x + w, size)] * size, 0)
        matrix[:, :, 0] = torch.stack([torch.linspace(y, y + h, size)] * size, 1)
        return matrix


    def _get_A_matrix(self, base, moment, point):
        """
        Get A matrix
        base : base coordinates
        moment : moment coordinates
        point : base rect's position (x, y, w, h)
        """
        x1, y1, w1, h1 = point

        diag_len = (torch.sqrt((w1.float() ** 2) + (h1.float() ** 2))) / 7

        A_matrix = self._get_normalized_distance(base, moment, diag_len)
        return A_matrix

    def _get_normalized_distance(self, base, moment, diaglen):
        """
        Get normalized distance
        base : base coordinates
        moment : moment coordinates
        diaglen : base rect's diangonal length for normalization
        """
        size = base.shape[0] * base.shape[1]

        base_x_matrix = base[:, :, 1]
        base_y_matrix = base[:, :, 0]

        moment_x_matrix = moment[:, :, 1]
        moment_y_matrix = moment[:, :, 0]

        # to compute the pairwise distance
        dist_x_matrix = torch.mm(base_x_matrix.view(-1, 1), torch.ones((1, size))) - torch.mm(torch.ones((size, 1)),
                                                                                              moment_x_matrix.view(1,
                                                                                                                   -1))
        dist_y_matrix = torch.mm(base_y_matrix.view(-1, 1), torch.ones((1, size))) - torch.mm(torch.ones((size, 1)),
                                                                                              moment_y_matrix.view(1,
                                                                                                                   -1))

        dist_matrix = torch.sqrt(dist_x_matrix ** 2 + dist_y_matrix ** 2) / (diaglen / 7)
        A_matrix = torch.zeros((dist_matrix.shape))
        A_matrix[dist_matrix < self.args.threshold] = 1.
        A_matrix[dist_matrix >= self.args.threshold] = 0.

        return A_matrix

    def __len__(self):
        return len(self.samples)