import pdb

import torch
import glob
import torchvision

names = glob.glob("yamaha/annotations/training/*")
cls_count = torch.zeros(9)
for n in names:
    img = torchvision.io.read_image(n)
    c_num, c_count = img.unique(return_counts=True)
    for idx in range(len(c_num)):
        cls_count[c_num[idx].item()] += c_count[idx]

print(cls_count)
print(cls_count / cls_count.max())
print(1 / (cls_count / cls_count.max()))


