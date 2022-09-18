import torch
import torch.nn as nn

# This GAN is created using a tutorial from youtube: https://youtu.be/IZtv9s_Wx9I?list=LL
# Which is in turn based on (a / the original) paper on GAN:s (paper.pdf in the repo)
# All paramaters and constants are taken from the paper 
# The original code has an MIT license

# nn.module is a base class for all neural network models in pytorch
# The models should subclass this class
class Discriminator(nn.module):
	def __init__(self, channels_img, features_d):
		super(Discriminator, self).__init__()
		self.disc = nn.Sequential(
			# Kernel size, stride and padding is inferred from the paper
			# Input: N x channels_img x 64 x 64
			nn.Conv2d(
				channels_img, features_d, kernel_size=4, stride=2, padding=1
			), # 32 x 32
			nn.LeakyReLU(0.2),
			self._block(features_d, features_d*2, 4, 2, 1), # 16 x 16
			self._block(features_d*2, features_d*4, 4, 2, 1), # 8 x 8
			self._block(features_d*4, features_d*8, 4, 2, 1), # 4 x 4
			# Output is a single value (1 x 1) representing if the image is fake or real
			nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
			nn.Sigmoid(), # The value is passed through a sigmoid layer to ensure that it is between 0 and 1
		)

	def _block(self, in_channels, out_channels, kernel_size, stride, padding):
		# Allows for sequential processing of data, (layers?)
		return nn.Sequential( 
			# Two dimensional convolution, bias is not used since batch normalization is used
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
			# Normalizing layers inputs to make training faster and more stable 
			nn.BatchNorm2d(out_channels),
			# Like regular ReLu but it has a slope for negative values instead of a constant value
			nn.LeakyReLU(0.2)
		)
		
	# To move through the layers?
	def forward(self, x):
		return self.disc(x) 