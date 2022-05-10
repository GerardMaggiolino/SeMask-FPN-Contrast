import pdb

import torch
import numpy as np
import torchvision
import glob
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from mmcv import Config
from mmseg.models import build_segmentor


def get_model(checkpoint_path, config_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    classes = ckpt["meta"]["CLASSES"]
    palette = ckpt["meta"]["PALETTE"]
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)
    for k in model.state_dict():
        if k not in state_dict:
            raise RuntimeWarning("Loading weights, missing {} weight".format(k))
    model.load_state_dict(state_dict, strict=False)
    model.classes = classes
    model.palette = torch.tensor(palette).float() / 255
    return model

def get_random_images(n=5):
    form = torchvision.transforms.Normalize(
        mean=[109.6286, 114.4169, 108.3678],
        std=[18.9286, 18.3765, 17.6692],
    )


    names = random.choices(glob.glob("data/yamaha/images/validation/*"), k=n)
    images = []
    labels = []
    for n in names:
        img = form(torchvision.io.read_image(n).float())
        images.append(img)
        lab = torchvision.io.read_image(n.replace("images", "annotations").replace("jpg", "png"))
        labels.append(lab)
    images = torch.stack(images).cuda()
    labels = torch.stack(labels).cuda()
    return images, labels


def tsne_vis(call_func, model, save_name, img, lab):
    torch.random.manual_seed(0)
    np.random.seed(0)
    model.cuda()
    model.eval()

    with torch.no_grad():
        feats = call_func(img, rescale=True)
    feats = feats.permute((0, 2, 3, 1)).reshape(-1, feats.shape[1])
    lab = lab.permute((0, 2, 3, 1)).reshape(-1)

    inds = []
    rep = lab.unique()
    # Removing background
    for cls_idx in rep[1:]:
        cls_inds_to_choose = torch.where(lab == cls_idx)[0]
        selected = cls_inds_to_choose[torch.randperm(len(cls_inds_to_choose))][:30].cpu().tolist()
        inds.extend(selected)

    inds = torch.tensor(inds).long()
    feats = feats[inds]
    # Removing background
    lab = lab[inds] - 1

    reduced_feats = TSNE().fit_transform(feats.cpu().numpy())

    plt.scatter(reduced_feats[:, 0], reduced_feats[:, 1], c=model.palette[lab.long()])
    plt.title(save_name)
    plt.savefig(save_name + ".png")
    plt.clf()




def main():
    img, lab = get_random_images()
    # Best scoring weights
    vanilla_cfg = "configs/semask_swin/custom/semfpn_semask_swin_small_yamaha.py"
    vanilla_pt = "work_dirs/semfpn_semask_swin_small_yamaha/latest.pth"
    model = get_model(vanilla_pt, vanilla_cfg)

    tsne_vis(model.inference_features, model, "Vanilla SeMask TSNE for Different Classes", img, lab)


    contrast_cfg = "configs/semask_swin/custom/contrastive_semfpn_semask_swin_small_yamaha.py"
    contrast_pt = "work_dirs/contrastive_semfpn_semask_swin_small_yamaha/latest.pth"
    model = get_model(contrast_pt, contrast_cfg)
    tsne_vis(model.inference_features_cluster, model, "Contrastive SeMask TSNE for Different Classes", img, lab)



main()