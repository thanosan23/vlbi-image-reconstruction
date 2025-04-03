import numpy as np
from piqa import SSIM
import pytorch_ssim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import ehtim as eh
from ehtim_helper import modify_telescope_positions
from torch.nn import functional as F


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.images = []
        for file in glob.glob(folder + '/*/*'):
            image = np.asarray(Image.open(file).convert('RGB').resize(
                (100, 100), Image.LANCZOS))
            self.images.append(image)
        self.images = np.array(self.images)
        np.random.shuffle(self.images)

        self.new_images = []
        self.outputs = []
        self.transform = transform

        eht = eh.array.load_txt('../arrays/EHT2017.txt')

        new_positions = []
        eht = modify_telescope_positions(eht, new_positions)

        for i in range(len(self.images)):
            gray_image = np.dot(self.images[i][..., :3], [
                0.2989, 0.5870, 0.1140])
            gray_image = gray_image - gray_image.min()
            gray_image = gray_image / (gray_image.max() + 1e-8)
            self.new_images.append(gray_image)
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


class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, channels, num_pool_layers, drop_prob):
        super().__init__()
        self.down_sample_layers = nn.ModuleList(
            [ConvBlock(in_channels, channels, drop_prob)])
        ch = channels
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, out_channels, kernel_size=1, stride=1),
            )
        )

    def forward(self, x):
        stack = []
        out = x

        for layer in self.down_sample_layers:
            out = layer(out)
            stack.append(out)
            out = nn.functional.avg_pool2d(
                out, kernel_size=2, stride=2, padding=0)

        out = self.conv(out)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            out = transpose_conv(out)

            if out.shape[-1] != downsample_layer.shape[-1]:
                out = nn.functional.pad(out, [0, 1, 0, 1], "reflect")

            out = torch.cat([out, downsample_layer], dim=1)
            out = conv(out)

        out = nn.functional.sigmoid(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x):
        return self.layers(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


folder = 'data'
transform = transforms.Compose([transforms.ToTensor()])
dataset = ImageDataset(folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "mps"
model = UNetModel(in_channels=1, out_channels=1, channels=64,
                  num_pool_layers=4, drop_prob=0.2).to(device)


class SSIMLoss(SSIM):
    def __init__(self):
        super().__init__(n_channels=1)

    def forward(self, x, y):
        return 1. - super().forward(x, y)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ssim_loss = SSIMLoss()

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def psnr_loss(self, pred, target):
        mse = self.mse_loss(pred, target)
        psnr = 10 * torch.log10(1.0 / mse)
        return -psnr

    def forward(self, pred, target):
        ssim_loss = 1 - self.ssim_loss(pred, target)

        psnr_loss = self.psnr_loss(pred, target)

        mse_loss = self.mse_loss(pred, target)

        total_loss = self.alpha * ssim_loss + \
            self.beta * psnr_loss + self.gamma * mse_loss
        return total_loss


# criterion = nn.MSELoss()
# criterion = SSIMLoss().to(device)
criterion = CombinedLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-5)


epochs = 200
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for dirty, clean in tqdm(dataloader):
        dirty, clean = dirty.to(device), clean.to(device)

        optimizer.zero_grad()
        pred = model(dirty)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'unet_image_3.pth')

model.eval()
dirty_image, test_data = next(iter(dataloader))
dirty_image, test_data = dirty_image.to(device), test_data.to(device)
pred = model(dirty_image).cpu().detach().numpy()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].set_title("Dirty Image")
ax[0].imshow(dirty_image[0][0].cpu().detach().numpy(), cmap="inferno")
ax[1].set_title("Original Image")
ax[1].imshow(test_data[0][0].cpu().detach().numpy(), cmap="inferno")
ax[2].set_title("Prediction")
ax[2].imshow(pred[0][0], cmap="inferno")
plt.show()
