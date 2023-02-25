

def convert_config(args, cfg):
    args.distributed_training = cfg.distributed_training
    args.dataset = cfg.dataset
    args.encoder = cfg.encoder
    args.work_dirs = cfg.work_dirs
    args.ppm_module = cfg.ppm_module
    args.training_parameters = cfg.training_parameters
    args.resume = cfg.resume
    args.devive = cfg.device
    args.lr = cfg.lr
    return args