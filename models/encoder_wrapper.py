from models.resnet import ResNetWrapper


class EncoderWrapper:
    def __init__(self, cfg):
        super(EncoderWrapper, self).__init__()
        self.backbone = cfg['backbone']
        self.dim1 = cfg['latent_dims_1']
        self.dim2 = cfg['latent_dims_2']
        self.replace_stride_with_dilation = cfg['replace_stride_with_dilation']

        property(self.backbone,
                 self.replace_stride_with_dilation,
                 self.dim1,
                 self.dim2)

    def __call__(self):
        return ResNetWrapper(self)