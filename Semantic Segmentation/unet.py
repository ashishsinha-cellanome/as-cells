from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

from typing import Final, List, Tuple, Optional, Dict

# default input size to the model
INPUT_IMAGE_HEIGHT: Final[int] = 1024
INPUT_IMAGE_WIDTH: Final[int] = 1024

class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)
    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
    def __init__(self, channels):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encoder_blocks = ModuleList([Block(channels[i], channels[i + 1]) 
                                          for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []
        # loop through the encoder blocks
        for block in self.encoder_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return block_outputs
    
class Decoder(Module):
    def __init__(self, channels):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2) 
                                   for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([Block(channels[i], channels[i + 1]) 
                                      for i in range(len(channels) - 1)])
    def forward(self, x, encoded_features):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            cropped_enc_features = self.crop(encoded_features[i], x)
            x = torch.cat([x, cropped_enc_features], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    def crop(self, encoded_features, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encoded_features = CenterCrop([H, W])(encoded_features)
        # return the cropped features
        return encoded_features
    
class UNet(Module):
    def __init__(self, encoder_channels=(3, 16, 32, 64),
                 decoder_channels=(64, 32, 16),
                 num_classes=2, retain_dim=True,
                 out_size=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decoder_channels[-1], num_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size
    
    def forward(self, x):
        # grab the features from the encoder
        encoded_features = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decoded_features = self.decoder(encoded_features[::-1][0], encoded_features[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        seg_map = self.head(decoded_features)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retain_dim:
            seg_map = F.interpolate(seg_map, self.out_size)
        # return the segmentation map
        # the training script is expecting a dictionary with key 'out' for the segmentation map
        out = {}
        out['out'] = seg_map
        return out
