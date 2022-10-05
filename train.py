from cProfile import label
from configparser import Interpolation
from datetime import datetime
from http.client import LENGTH_REQUIRED
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt
import cv2 
import os
import PIL
import numpy as np
import matplotlib.image as img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determining GPU or CPU
LEARNING_RATE = 2e-4 # How drastic the change of the model is between each epoch
# The number of samples that will be used in each epoch of training
# 2 x 64 to train gen and disc in parallell?
BATCH_SIZE = 128 
IMAGE_SIZE = 64
CHANNELS_IMG = 3 # Grayscale/RGB/RGBA etc.
Z_DIM = 100 # Dimension of the initial uniform distribution from the paper
NUM_EPOCHS = 11 # Number of training cycles
FEATURES_DISC = 64 
FEATURES_GEN = 64

# For plotting
loss_data_gen = []
loss_data_disc = []

# Resizes the training data, makes it into a tensor and normalizes the pixel values
transforms = transforms.Compose(
	[
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
	]
)

# Data loading
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,
#                        download=True)


dataset = datasets.CelebA(root="dataset/", split='train', transform=transforms,
                       download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device) # Sending the generator to the device
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device) # Sending the discriminator to the device
initialize_weights(gen) # Initialize weights for all operations in the generator
initialize_weights(disc) # Initialize weights for all operations in the discriminator

# Optimizers for the weights? Uses Adam algorithm?
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss() # 

# To see progression
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device) 
# Visualization?
writer_real = SummaryWriter(f"logs/real" + datetime.now().strftime("%Y%m%d-%H%M%S") + "Loss MSE" + "Dataset MNIST" + "Epochs " + str(NUM_EPOCHS))
writer_fake = SummaryWriter(f"logs/fake" + datetime.now().strftime("%Y%m%d-%H%M%S") + "Loss MSE" + "Dataset MNIST" + "Epochs " + str(NUM_EPOCHS))
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
	# Since GAN:s are unsupervised we don't use the image labels in training
	for batch_idx, (real, _) in enumerate(loader):
		real = real.to(device) # Send real training data to device
		noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
		fake = gen(noise) # Fake images generated as noise

		# Train Discriminator wants to maximize: log(D(x)) + log(1 - D(G(z)))
		disc_real = disc(real).reshape(-1) # To get a singular value instead of N x 1 x 1 x 1
		loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
		disc_fake = disc(fake).reshape(-1)
		loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
		loss_disc = (loss_disc_real + loss_disc_fake) / 2
		disc.zero_grad()
		loss_disc.backward(retain_graph=True)
		opt_disc.step()

		# Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))
		output = disc(fake).reshape(-1)
		loss_gen = criterion(output, torch.ones_like(output))
		gen.zero_grad()
		loss_gen.backward()
		opt_gen.step()

		if batch_idx % 100 == 0:
			print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

			

			loss_data_disc.append(loss_disc.cpu().detach().numpy())
			loss_data_gen.append(loss_gen.cpu().detach().numpy())

			with torch.no_grad():
				fake = gen(fixed_noise)
				img_grid_real = torchvision.utils.make_grid(
					real[:32], normalize=True
				)
				img_grid_fake = torchvision.utils.make_grid(
					fake[:32], normalize=True
				)
				writer_real.add_image("Real", img_grid_real, global_step=step)
				writer_fake.add_image("Fake", img_grid_fake, global_step=step)
			step += 1

			if epoch == 10:
				for i in range(32):
					np_fake = fake.cpu().numpy()
					np_real = real.cpu().numpy()

					path_fake = os.path.join('D:\DCGAN\CELEBA_BCE_10_F', str(batch_idx) + '_' + str(i) + '_F' + '.jpg')
					path_real = os.path.join('D:\DCGAN\CELEBA_BCE_10_R', str(batch_idx) + '_' + str(i) + '_R' + '.jpg')

					# # MNIST

					# cv2.imwrite(path_fake, np.multiply(np_fake[i, 0, :, :], 255).astype(np.uint8))
					# cv2.imwrite(path_real, np.multiply(np_real[i, 0, :, :], 255).astype(np.uint8))

					# CELEBA

					# Transpose to valid image dimensions
					np_fake_t = np.transpose(np_fake[i, :, :, :], (1, 2, 0))
					np_real_t = np.transpose(np_real[i, :, :, :], (1, 2, 0))

					# Normalize image values to [0, 1]
					np_fake_n = (np_fake_t - np.min(np_fake_t)) / np.ptp(np_fake_t)
					np_real_n = (np_real_t - np.min(np_real_t)) / np.ptp(np_real_t)

					# Save images
					img.imsave(path_fake, np_fake_n)
					img.imsave(path_real, np_real_n)

# Plotting
plt.figure()
plt.plot(loss_data_disc, label='Loss Discriminator')
plt.plot(loss_data_gen, label='Loss Generator')
plt.legend(loc="upper right")
plt.title(str(NUM_EPOCHS) + ' EPOCHS, CELEBA, BCE')
plt.ylabel('Loss')
plt.figure()
np_grid = img_grid_fake.cpu().numpy()
plt.axis('off')
plt.imshow(np.transpose(np_grid, (1, 2, 0)), interpolation='nearest')
plt.show()