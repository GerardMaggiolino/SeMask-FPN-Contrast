_base_ = [
    '../../_base_/models/semfpn_semask_swin.py', '../../_base_/datasets/semask/cityscapes_768x768.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        num_cls=19,
        sem_window_size=12,
        num_sem_blocks=[1,1,1,1],
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=19,
        cate_w=0.4
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-5,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 2 GPUs with 8 images per GPU
data=dict(samples_per_gpu=2)
