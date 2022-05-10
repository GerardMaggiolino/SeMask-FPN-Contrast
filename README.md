# Contrastive SeMask FPN


### Background 
This repo includes modifications to the [SeMask-FPN repository](https://github.com/Picsart-AI-Research/SeMask-Segmentation) to implement a variation of [Regional Contrastive Loss](https://shikun.io/projects/regional-contrast). It's part of a Visual Learning and Recognition (16-824) final project at Carnegie Mellon University. 

The original project focuses on self-supervised multimodality cost-learning for autonomous robotic navigation through rough terrain, but uses a patch-based vision system with high latency. Sensor measurements are associated with the visual region which the robot is actively traversing. Cost is inferred from sensor feedback and a network is trained offline to regress cost from the region's appearance (further details are on hold due to pending publication). However, dense, per-pixel embeddings from images can be used to propagate costs based on visual similarity in an online fashion. 

Cost assigned to an image patch can be related to the full image through similarity of dense embeddings. While the pre-class layer features of any off-the-shelf pre-trained segmentation model will be fairly discriminative, recent work has shown contrastive loss in segmentation models can disentangle and regularize the embedding space - additionally, explicitly L2-normalized features provide a bounded metric. The contrastive SeMask model is trained over the [YCOR dataset](https://theairlab.org/yamaha-offroad-dataset/) to learn task-specific discriminative features. 

### Training Setup 

- Install the requirements as stated in the original [SeMask-FPN repo](https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FPN) and run `pip install -e .` from the top level directory. 
- Download the [YCOR dataset](https://theairlab.org/yamaha-offroad-dataset/) to `data/yamaha`. 
- Run `python convert_labels.py` from the `data` directory (the paths are hardcoded for simplicity). This converts the YCOR dataset into an ADE20K format. 
- Download pre-trained SeMask-S FPN weights [from here](https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FPN#ade20k). 

After the set up, you can run 
```
python train.py configs/semask_swin/custom/semfpn_semask_swin_small_yamaha.py —load-from <path/to/pretrained/ade20k/weights>
```
to train the vanilla SeMask-S FPN on YCOR, or 
```
python train.py configs/semask_swin/custom/contrastive_semfpn_semask_swin_small_yamaha.py —load-from <path/to/pretrained/ade20k/weights>
```
to run the contrastive model. 

The core code of the contrastive model (that's new to SeMask) is located under `mmseg/models/decode_heads/` as `contrastive_semask_decode_head.py` and `contrastive_branch_fpn_head.py`. 
