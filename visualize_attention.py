from PIL import Image
import numpy as np
from io import BytesIO
import torch
from torch import nn
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from models.attention_model import AttentionModel
from data.base_dataset import get_transform

if __name__ == '__main__':
    load_path = "checkpoints/summer2winter_CUT/latest_net_D.pth"
    # img_path = "datasets/s2wy_test/testA/2010-09-07 12_23_20.jpg"
    img_path = "datasets/s2wy_test/testA/2012-07-08 16_40_31.jpg"
    # img_path = "datasets/s2wy_test/testB/2010-11-01 15_58_51.jpg"
    with torch.cuda.device(0):
        img = Image.open(img_path)
    # plt.imshow(img)

        opt = TrainOptions().parse()
        opt.no_flip = True
        transform=get_transform(opt)

    # preprocessed = transforms.compose([T.resize(256), T.toTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device('cuda:0')
    # net = networks.define_D(3, 64, "basic", 3, 'instance', 'xavier', 0.02, False, [0])
    # if isinstance(net, torch.nn.DataParallel):
    #     net = net.module
    attention_model = AttentionModel(opt)
    state_dict = torch.load(load_path, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    attention_model.netD.load_state_dict(state_dict)

    attention_model.layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    attention_model.input_imgs = transform(img).unsqueeze(0).cuda()


    attention_model.eval()
    with torch.no_grad():
        maps = attention_model.generate_attention()

    counter = 0
    for image_map in maps:
        print(image_map.shape)
        image_map = torch.squeeze(image_map)
        print(image_map.shape)
        plt.imshow(image_map.cpu(), interpolation='bicubic')
        plt.savefig("visualize_am2/attention_map_" + str(counter) + ".png")
        counter += 1
    
        