#import torch
import argparse
#import torchvision.models
import configs.clld as cfg
from models.encoder_wrapper import EncoderWrapper


def convertor(model, last_model):
    if last_model is not None:
        print('Converting...')
        for new_n, new_p in model.named_parameters():

            for i in last_model['state_dict']:
                if (i == 'module.base_encoder.' + new_n) and (new_p.shape == last_model['state_dict'][i].shape):
                    new_p.data = last_model['state_dict'][i].data
                    print(i, 'is converted to torchvision')
    torch.save(model.state_dict(), 'published_CLLD_checkpoint.pth')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Publish weights')

    parser.add_argument(
        '--encoder', default='resnet50',
        help='choose an encoder for training. ["resnet18", "resnet34", '
             '"resnet50", "resnet101", "resnet152", "resnext50_32x4d", '
             '"resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"]')

    parser.add_argument (
        '--checkpoint', type=str, help='path to checkpoint file')

    args = parser.parse_args()

    weight = torch.load(args.checkpoint, map_location='cpu')

    cfg.encoder['backbone'] = args.encoder

    resnet = EncoderWrapper(cfg.encoder)().to('cpu')

    resnet = convertor(resnet, weight)