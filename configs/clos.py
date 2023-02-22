dataset = dict(
    path='/media/ali/ssd/Imagenet/train',
    work_dir= 'work_dir',
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
    replace_stride_with_dilation=[False, True, True],
)

work_dirs = dict(
    path='experiments',
    ckpt_dir='checkpoints',
    log_dir='runs'
)

ppm_module = dict(
    input_channels=256,
    output_channels=256
)

training_parameters = dict(
    num_epochs=100,
    batch_size=1024,
    init_lr=1,
    weight_decay=15e-7,
    workers=16,
    optim_eps=1e-8,
    temperature=0.3,
    encoder_momentum=0.99,
    grouping_threshold=0.7,
    alpha=3,
    out_scale=0.001,
    coeff_lamda=1,
    coeff_beta=2,
    coeff_Xi=1
)

resume = dict(
    start_epoch=0,
    ckpt_path=None
)

distributed_training = dict(
    world_size=1,
    rank=0,
    env_ip='tcp://10.132.136.34:5019',
    backend='nccl',
    gpus_idx=[0, 1],
)