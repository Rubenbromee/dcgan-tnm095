from http.client import LENGTH_REQUIRED
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determining GPU or CPU
LEARNING_RATE = 2e-4 # How drastic the change of the model is between each epoch
# The number of samples that will be used in each epoch of training
# 2 x 64 to train gen and disc in parallell?
BATCH_SIZE = 128 
IMAGE_SIZE = 64
CHANNELS_IMG = 1 # Grayscale/RGB/RGBA etc.
Z_DIM = 100 # Dimension of the initial uniform distribution from the paper
NUM_EPOCHS = 5 # Number of training cycles
FEATURES_DISC = 64 
FEATURES_GEN = 64

# Resizes the training data, makes it into a tensor and normalizes the pixel values
transforms = transforms.Compose(
	[
		transforms.Resize(IMAGE_SIZE, IMAGE_SIZE),
		transforms.ToTensor(),
		transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
	]
)

# Data loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device) # Sending the generator to the device
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device) # Sending the discriminator to the device
initialize_weights(gen) # Initialize weights for all operations in the generator
initialize_weights(disc) # Initialize weights for all operations in the discriminator

# Optimizers for the weights? Uses Adam algorithm?
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss() # Binary cross entropy loss function

# To see progression
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device) 
# Visualization?
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
	for batch_idx, (real, _) in enumerate(loader):
		real = real.to(device) # Send real training data to device
		noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)

		# Train Discriminator max log(D(x)) + log(1 - D(G(z)))
		disc_real = disc(real).reshape(-1) # To get a singular value instead of N x 1 x 1 x 1
		loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
		disc_fake = disc(fake).reshape(-1)
		loss_disc_fake = criterion(fake, torch.zeros_like(disc_fake))
		loss_disc = (loss_disc_real + loss_disc_fake) / 2
		disc.zero_grad()
		loss_disc.backward(retain_graph=True)
		opt_disc.step()

		# Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))

