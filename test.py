import torch
from PIL import Image
from torchvision import transforms
from core.model import FuseModel
from core.utils.config import load_config


if __name__ == '__main__':
    config = load_config('config/GAN_G1_D2_3Conv.yaml')
    GAN_Model = FuseModel(config, val=True)
    vis_img = Image.open('demo/test_vis.jpg')
    inf_img = Image.open('demo/test_inf.jpg')
    trans = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])
    vis_img = trans(vis_img)
    inf_img = trans(inf_img)
    data = {'Vis': vis_img.unsqueeze(0), 'Inf': inf_img.unsqueeze(0)}
    GAN_Model.Generator.load_state_dict(torch.load('weights/GAN_G1_D2_3Conv/generator/generator_57.pth'))
    GAN_Model.eval()
    Generator_feats, _, _ = GAN_Model(data)
    untrans = transforms.Compose([transforms.ToPILImage()])

    img = untrans(Generator_feats['Generator_1'][0])
    print(img.size)
    img.save('./demo/test_result.jpg')
