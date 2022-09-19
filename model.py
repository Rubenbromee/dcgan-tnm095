import torch
import torch.nn as nn

# This GAN is created using a tutorial from youtube: https://youtu.be/IZtv9s_Wx9I?list=LL
# Which is in turn based on (a / the original) paper on GAN:s (paper.pdf in the repo)
# All paramaters and constants are taken from the paper 
# The original code has an MIT license

# nn.module is a base class for all neural network models in pytorch
# The models should subclass this class
class Discriminator(nn.Module):
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

class Generator(nn.Module):
	def __init__(self, z_dim, channels_img, features_g):
		super(Generator, self).__init__()
		self.gen = nn.Sequential(
			# Input: N x z_dim x 1 x 1
			self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4 x 4
			self._block(features_g*16, features_g*8, 4, 2, 1), # 8 x 8
			self._block(features_g*8, features_g*4, 4, 2, 1), # 16 x 16
			self._block(features_g*4, features_g*2, 4, 2, 1), # 32 x 32
			# Final image is 64 x 64 pixels
			nn.ConvTranspose2d(
				features_g*2, channels_img, kernel_size=4, stride=2, padding=1
			),
			nn.Tanh(), # To adjust pixel values to the span [-1, 1]
		)
	
	def forward (self, x):
		return self.gen(x)
	
	def _block(self, in_channels, out_channels, kernel_size, stride, padding):
		return nn.Sequential(
			# Transposed convolution (upscales the image?)
			# Again, no bias since we use batch normalization between layers
			nn.ConvTranspose2d(
				in_channels,
				out_channels,
				kernel_size,
				stride,
				padding,
				bias=False
			),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(), # Regular ReLU, as in the paper
		)

# Set up initial weights for disc/gen as a normal distribution
def initialize_weights(model):
	for m in model.modules():
		if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
			nn.init.normal_(m.weight.data, 0.0, 0.02)

# Test to make sure the shapes of the gen/disc match up with those in the paper
def test():
	N, in_channels, H, W = 8, 3, 64, 64
	z_dim = 100
	x = torch.randn(N, in_channels, H, W) # Random image noise
	disc = Discriminator(in_channels, 8)
	initialize_weights(disc)
	assert disc(x).shape == (N, 1, 1, 1)
	gen = Generator(z_dim, in_channels, 8)
	z = torch.randn((N, z_dim, 1, 1)) # Latent noise
	assert gen(z).shape == (N, in_channels, H, W)
	print("Success")

# test()