import pdb

import torch
import glob
import torchvision

names = glob.glob("yamaha/images/training/*")
rgb = []
for n in names:
    rgb.append(torchvision.io.read_image(n).to(torch.float).mean(axis=[1, 2]))

pdb.set_trace()
rgb = torch.stack(rgb)
print("std,", rgb.std(dim=0))
print("mean,", rgb.mean(dim=0))

