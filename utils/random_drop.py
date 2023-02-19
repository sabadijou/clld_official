import numpy as np
import itertools
import random


class RandomDrop:
    def __init__(self, lambda_=14,
                 gamma_=14,
                 information_loss=0.3,
                 height=224,
                 weight=224):
        super(RandomDrop, self).__init__()

        self.lambda_ = lambda_

        self.gamma_ = gamma_

        self.IF = information_loss

        self.height = height

        self.weight = weight

    def __call__(self, x):
        coor_list = self._make_coordinates()
        num_patches = self.cal_remove_patches(len_all_coordinates=len(coor_list))

        if len(x.shape) == 2:
            return x
        if x.shape[2] == 4:
            x = x[:, :, :3]

        random_patch = np.random.randint(0, 256, size=(self.gamma_, self.lambda_, 3))

        for i in range(num_patches):
            coordinate = random.choice(coor_list)
            coor_list.remove(coordinate)
            x[coordinate[0]:coordinate[0] + self.gamma_,
              coordinate[1]:coordinate[1] + self.lambda_, ...] = random_patch

        return x

    def _make_coordinates(self):
        coor_list_h = [i for i in range(0, self.height, self.gamma_)]
        coor_list_w = [i for i in range(0, self.weight, self.lambda_)]

        return list(itertools.combinations(coor_list_h, 2)) + \
            list(itertools.combinations(coor_list_w.__reversed__(), 2)) + \
            [(i, i) for i in range(0, 224, self.lambda_)]

    def cal_remove_patches(self, len_all_coordinates):
        number_of_patches = int(len_all_coordinates * self.IF)

        return number_of_patches
