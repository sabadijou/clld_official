import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import cv2
####################################################################
patch_size = 14
image_size = (224, 224)
information_loss = 0.3
start_points = [i for i in range(0, 224, patch_size)]
coor_list = list(itertools.combinations(start_points, 2)) + \
            list(itertools.combinations(start_points.__reversed__(), 2)) + \
            [(i, i) for i in range(0, 224, patch_size)]

number_of_patches = int(len(coor_list) * information_loss)
####################################################################
patch_dir = r'C:\Users\asus\Desktop\lane\nat_barrat'
image_lists = os.listdir(patch_dir)
####################################################################
for img in image_lists:
    print(img)
    start_points = [i for i in range (0, 224, patch_size)]
    coor_list = list (itertools.combinations (start_points, 2)) + \
                list (itertools.combinations (start_points.__reversed__ (), 2)) + \
                [(i, i) for i in range (0, 224, patch_size)]
    image = Image.open(os.path.join(patch_dir, img))
    image = image.resize(image_size)
    image = np.asarray(image)
    if len(image.shape) == 2:
        continue
    if image.shape[2] == 4:
        image = image[:, :, :3]
    ####################################################################
    random_patch = np.random.randint(0, 256, size=(patch_size, patch_size, 3))
    for i in range(number_of_patches):
        coordinate = random.choice(coor_list)
        coor_list.remove(coordinate)
        image[coordinate[0]:coordinate[0] + patch_size, coordinate[1]:coordinate[1] + patch_size, ...] = random_patch

    cv2.imwrite(os.path.join(r'C:\Users\asus\Desktop\lane\converted', img), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# plt.imshow(image)
# plt.axis('off')
# plt.
# plt.savefig(r'patch.jpg')
# plt.show()