import torch
import torchvision.models


def convertor(model, last_model):
    if last_model is not None:
        print('Converting...')
        for new_n, new_p in model.named_parameters():

            for i in last_model['state_dict']:
                if (i == 'module.base_encoder.' + new_n) and (new_p.shape == last_model['state_dict'][i].shape):
                    new_p.data = last_model['state_dict'][i].data
                    print(i, 'is converted to torchvision')
    torch.save(model.state_dict(), 'Resnet50_no_occlusion_masking.pth')
    return model


if __name__ == '__main__':
    weight = torch.load(r'../Documents/Resnet50_no_occlusion_masking.pth.tar',
                        map_location='cpu')
    resnet = torchvision.models.resnet50().to('cpu')
    resnet = convertor(resnet, weight)