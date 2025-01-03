import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import ehtim as eh

from ehtim_helper import modify_telescope_positions


class Galaxy10Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        with h5py.File(hdf5_file, 'r') as f:
            self.images = np.array(f['images'])
            self.labels = np.array(f['ans'])
        self.images = self.images[:3000]
        self.new_images = []
        self.outputs = []
        self.transform = transform

        eht = eh.array.load_txt('../arrays/EHT2017.txt')

        new_positions = []
        eht = modify_telescope_positions(eht, new_positions)

        for i in range(len(self.images)):
            self.new_images.append(np.dot(self.images[i][..., :3], [
                0.2989, 0.5870, 0.1140]))  # convert to grayscale
            eht_fov = 200 * eh.RADPERUAS
            eht_size = self.new_images[-1].shape[0]
            im = eh.image.Image(
                self.new_images[-1],
                psize=eht_fov / eht_size,
                ra=0.0,
                dec=0.0,
                rf=230e9,
                source='SyntheticImage'
            )
            tint_sec = 60
            tadv_sec = 600
            tstart_hr = 0
            tstop_hr = 24
            bw_hz = 4.e9

            obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr,
                             bw_hz, sgrscat=False, ampcal=True, phasecal=True,
                             ttype='direct', verbose=False)

            fov = 200 * eh.RADPERUAS
            dim = obs.dirtyimage(self.new_images[-1].shape[0], fov).imarr()
            self.outputs.append(dim)
            print("Done processing image", i+1, "out of", len(self.images))
        self.outputs = np.array(self.outputs)
        self.new_images = np.array(self.new_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.new_images[idx]
        image = Image.fromarray(image)
        out = Image.fromarray(self.outputs[idx])
        if self.transform:
            image = self.transform(image)
            dim = self.transform(out)
        return dim, image


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, input):
        return self.layers(input)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input):
        return self.layers(input)


class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, channels, num_pool_layers,
                 drop_prob):
        super().__init__()
        self.down_sample_layers = nn.ModuleList(
            [ConvBlock(in_channels, channels, drop_prob)])
        ch = channels
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input):
        stack = []
        output = input

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = nn.functional.avg_pool2d(
                output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if sum(padding) != 0:
                output = nn.functional.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


hdf5_file = 'galaxy10.h5'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
dataset = Galaxy10Dataset(hdf5_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = UNetModel(in_channels=1, out_channels=1, channels=64,
                  num_pool_layers=4, drop_prob=0.2).to("mps")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

epochs = 250
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader):
        dims, images = batch
        dims = dims.to("mps")
        images = images.to("mps")

        outputs = model(dims)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
    )

torch.save(model.state_dict(), 'unet_galaxy10_2.pth')

model.eval()
dirty_image, test_data = next(iter(dataloader))
pred = model(dirty_image.to("mps"))

fig, ax = plt.subplots(2, 2)
ax[0, 0].set_title("Dirty Image")
ax[0, 0].imshow(dirty_image[0][0].cpu().detach().numpy())
ax[0, 1].set_title("Original Image")
ax[0, 1].imshow(test_data[0][0].cpu().detach().numpy())
ax[1, 0].set_title("Prediction")
ax[1, 0].imshow(pred[0][0].cpu().detach().numpy())
plt.show()
