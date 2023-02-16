dataset = dict(
    path='/media/ali/ssd/Imagenet/train',
    sample_size=224,
    seed=42,
    information_loss=0.3,
    _lambda=14,
    _gamma=14
)

encoder = dict(
    backbone='resnet50',
    latent_dims_1=2048,
    latent_dims_2=256,
)

work_dirs = dict(
    path='experiments',
    ckpt_dir='checkpoints',
    log_dir='runs'
)

training_parameters = dict(
    num_epochs=100,
    batch_size=1024,
    init_lr=1,
    weight_decay=1e-5,
    workers=16,
    temperature=0.3,
    encoder_momentum=0.99,
    grouping_threshold=0.7,
    alpha=3,
    out_scale=0.001
)

distributed_training = dict(
    world_size=1,
    rank=0,
    env_ip='tcp://10.132.136.34:5019',
    backend='nccl',
    gpu_idx=[0, 1],
)