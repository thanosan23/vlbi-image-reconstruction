import skimage.metrics
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


def generate_dirty(image):
    eht = eh.array.load_txt('arrays/EHT2017.txt')

    new_positions = []
    eht = modify_telescope_positions(eht, new_positions)

    image = np.array(image)
    # image = image.unsqueeze(0)
    test_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    eht_fov = 200 * eh.RADPERUAS
    eht_size = test_image.shape[0]
    im = eh.image.Image(
        test_image,
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

    obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                     sgrscat=False, ampcal=True, phasecal=True,
                     ttype='direct', verbose=True)

    fov = 200 * eh.RADPERUAS
    dim = obs.dirtyimage(
        test_image.shape[0], fov).imarr()
    return dim


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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = UNetModel(in_channels=1, out_channels=1, channels=64,
                  num_pool_layers=4, drop_prob=0.2)

TEST_TYPE = "galaxy"

if TEST_TYPE == "galaxy":
    with h5py.File('galaxy10.h5', 'r') as f:
        images = np.array(f['images'])

    idx = np.random.randint(3000, images.shape[0])

    test_image = images[idx]

    dim = generate_dirty(test_image)

    test_image = Image.fromarray(test_image)
    test_image = transform(test_image)
    dim = Image.fromarray(dim)
    dim = transform(dim)
elif TEST_TYPE == "model":
    test_image = np.array(Image.open("models/RIAF.png").resize((69, 69)))
    dim = generate_dirty(test_image)

    test_image = Image.fromarray(test_image)
    test_image = transform(test_image)
    dim = Image.fromarray(dim)
    dim = transform(dim)
else:
    raise ValueError("Invalid TEST_TYPE")

model.load_state_dict(torch.load("unet_galaxy10.pth", weights_only=True))
model.eval()
pred = model(dim.unsqueeze(0))

pred = pred.detach().numpy()
test_image = test_image.detach().numpy()


def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2*normalized_input - 1


test_imag = normalize_negative_one(test_image)
pred = normalize_negative_one(pred)


print(test_image[0].max(), test_image[0].min())
print(pred[0].max(), pred[0].min())

# rpint SSIM and MSE between images
ssim = skimage.metrics.structural_similarity(
    test_image[0], pred[0][0], data_range=2)

mse = skimage.metrics.mean_squared_error(
    test_image[0], pred[0][0])

print(f"SSIM: {ssim}, MSE: {mse}")
# set size
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(8, 5)
ax[0].set_title("Ground Truth")
ax[0].imshow(test_image[0])
ax[1].set_title("Dirty Image")
ax[1].imshow(dim[0])
ax[2].set_title("Prediction")
ax[2].imshow(pred[0][0])
plt.show()
