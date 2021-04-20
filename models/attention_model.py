import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class AttentionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # set defaults
        parser.set_defaults(TODO)
        if is_train:
            parser.add_argument(TODO)
            parser.set_defaults(gan_mode="lsgan") 
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D']
        self.visual_names = ['input', 'output']
        self.model_names = ['D']
        # self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netD = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.layers = [2, 6, 10, 14]
        if self.isTrain:
            self.criterionLoss = networks.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers = [self.optimizer_D]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_imgs = input[0]
        self.image_classes = input[1]

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netD(self.input_imgs)  # generate output image given the input data_A

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer_D.zero_grad()   # clear network G's existing gradient
        
        self.loss_D = 0
        for i in range(self.opt.batch_size):
            self.loss_D += self.criterionLoss(self.output[i], self.image_classes[i]).mean()
        self.loss_D /= self.opt.batch_size
        self.loss_D.backward()
        self.optimizer_D.step()        # update gradients for network G

    def generate_attention(self):
        "Return list of channel-wise squared mean feature maps"
        # feat_maps = self.netD.attention_forward(self.input_imgs, self.layers)
        feat_maps = self.netD(self.input_imgs, self.layers, encode_only=True)
        for i in range(len(feat_maps)):
            print(feat_maps[i].shape)
            feat_maps[i] = feat_maps[i].pow(2).mean(1)
        
        for feat in feat_maps:
            print(feat.shape)
        return feat_maps
