import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = torch.device("mps")  # for mac


def parse_config_file(file_path):
    antenna_positions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.startswith("#") and line.strip():
                pos = line.strip().split(',')
                if len(pos) == 2:
                    e_offset = float(pos[0].strip())
                    n_offset = float(pos[1].strip())
                    antenna_positions.append((e_offset, n_offset))
    return np.array(antenna_positions)


def compute_baselines(antenna_positions):
    baselines = []
    for (i, pos1), (j, pos2) in combinations(enumerate(antenna_positions), 2):
        baseline = pos2 - pos1
        baselines.append(baseline)
    return np.array(baselines)


def calculate_uv_coordinates(baselines, hour_angle, declination, wavelength):
    h0 = np.radians(hour_angle)
    delta0 = np.radians(declination)

    rotation_matrix = np.array([
        [np.sin(h0), np.cos(h0), 0],
        [-np.sin(delta0) * np.cos(h0), np.sin(delta0)
         * np.sin(h0), np.cos(delta0)],
        [np.cos(delta0) * np.cos(h0), -np.cos(delta0)
         * np.sin(h0), np.sin(delta0)]
    ])

    baselines_3d = np.hstack((baselines, np.zeros((baselines.shape[0], 1))))
    uvw_coordinates = np.dot(baselines_3d, rotation_matrix.T) / wavelength
    return uvw_coordinates[:, :2]


def load_custom_sky_image(image_path, image_size):
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size, image_size))
    sky_image = np.array(img) / 255.0
    return sky_image


def generate_visibilities(sky_image, uv_coordinates):
    ft_image = fftshift(fft2(ifftshift(sky_image)))
    visibilities = []
    image_center = np.array(sky_image.shape) // 2
    for u, v in uv_coordinates:
        u = int(round(u)) + image_center[0]
        v = int(round(v)) + image_center[1]
        if 0 <= u < sky_image.shape[0] and 0 <= v < sky_image.shape[1]:
            visibilities.append(ft_image[u, v])
        else:
            visibilities.append(0)
    return np.array(visibilities), ft_image


def generate_dirty_image(visibilities, uv_coordinates, image_size):
    ft_dirty_image = np.zeros((image_size, image_size), dtype=complex)
    image_center = np.array((image_size // 2, image_size // 2))

    for (u, v), vis in zip(uv_coordinates, visibilities):
        u_idx = int(round(u)) + image_center[0]
        v_idx = int(round(v)) + image_center[1]
        if 0 <= u_idx < image_size and 0 <= v_idx < image_size:
            ft_dirty_image[u_idx, v_idx] = vis
            ft_dirty_image[-u_idx % image_size, -v_idx %
                           image_size] = np.conj(vis)

    dirty_image = np.abs(ifftshift(ifft2(fftshift(ft_dirty_image))))
    return dirty_image, ft_dirty_image


def generate_beam_image(uv_coordinates, image_size):
    ft_beam_image = np.zeros((image_size, image_size), dtype=complex)
    image_center = np.array((image_size // 2, image_size // 2))
    for u, v in uv_coordinates:
        u_idx = int(round(u)) + image_center[0]
        v_idx = int(round(v)) + image_center[1]
        if 0 <= u_idx < image_size and 0 <= v_idx < image_size:
            ft_beam_image[u_idx, v_idx] = 1
            ft_beam_image[-u_idx % image_size, -v_idx % image_size] = 1

    beam_image = np.abs(ifftshift(ifft2(fftshift(ft_beam_image))))
    return beam_image


def gaussian_clean_beam(beam_image):
    return gaussian_filter(beam_image, sigma=5)


def clean(dirty_image, beam_image, gamma=0.1, threshold=0.001, max_iterations=100):
    clean_components = np.zeros_like(dirty_image)
    residuals = dirty_image.copy()
    beam_center = np.array(beam_image.shape) // 2

    for _ in range(max_iterations):
        max_pos = np.unravel_index(np.argmax(residuals), residuals.shape)
        max_value = residuals[max_pos]

        if max_value < threshold:
            break

        shift_y, shift_x = max_pos[0] - \
            beam_center[0], max_pos[1] - beam_center[1]
        scaled_psf = gamma * max_value * \
            np.roll(np.roll(beam_image, shift_y, axis=0), shift_x, axis=1)

        clean_components += scaled_psf
        residuals -= scaled_psf

    clean_beam = gaussian_clean_beam(beam_image)
    clean_image = ifftshift(
        ifft2(fftshift(fft2(clean_components) * fft2(clean_beam))))
    return np.abs(clean_image), clean_components, residuals, clean_beam


class ImageDataset(Dataset):
    def __init__(self, dirty_images, clean_images):
        self.dirty_images = dirty_images
        self.clean_images = clean_images

    def __len__(self):
        return len(self.dirty_images)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.dirty_images[idx],
                         dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.clean_images[idx],
                         dtype=torch.float32).unsqueeze(0),
        )


def generate_dataset(images):
    clean_images = []
    dirty_images = []
    for clean_image in images:
        sky_image = load_custom_sky_image(clean_image, image_size)
        visibilities, model_fft = generate_visibilities(
            sky_image, uv_coordinates)
        dirty_image, observed_fft = generate_dirty_image(
            visibilities, uv_coordinates, image_size)
        clean_images.append(sky_image)
        dirty_images.append(dirty_image)
    return clean_images, dirty_images


class ImageReconstructionNet(nn.Module):
    def __init__(self):
        super(ImageReconstructionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.tanh(self.conv6(x))
        return x


def train_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for dirty, clean in dataloader:
            dirty, clean = dirty.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(dirty)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


def plot_all_results(sky_image, uv_coordinates, beam_image, model_fft,
                     observed_fft, dirty_image, clean_image, cnn_image):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1)
    plt.imshow(sky_image, cmap='hot', origin='lower')
    plt.colorbar(label="Amplitude")
    plt.title("Model Image (Sky)")

    plt.subplot(3, 3, 2)
    plt.imshow(beam_image, cmap='hot', origin='lower')
    plt.colorbar(label="Amplitude")
    plt.title("Beam Image (PSF)")

    plt.subplot(3, 3, 3)
    plt.plot(uv_coordinates[:, 0], uv_coordinates[:, 1], 'bo', markersize=2)
    plt.plot(-uv_coordinates[:, 0], -uv_coordinates[:, 1], 'ro', markersize=2)
    plt.xlabel("u (wavelengths)")
    plt.ylabel("v (wavelengths)")
    plt.title("UV Coverage")
    plt.grid(True)
    plt.axis("equal")

    plt.subplot(3, 3, 4)
    plt.imshow(np.log(np.abs(model_fft)), cmap='hot', origin='lower')
    plt.colorbar(label="Log Amplitude")
    plt.title("Model FFT")

    plt.subplot(3, 3, 5)
    plt.imshow(np.log(np.abs(observed_fft) + 1e-10),
               cmap='hot', origin='lower')
    plt.colorbar(label="Log Amplitude")
    plt.title("Observed FFT")

    plt.subplot(3, 3, 6)
    plt.imshow(dirty_image, cmap='hot', origin='lower')
    plt.colorbar(label="Amplitude")
    plt.title("Dirty Image")

    plt.subplot(3, 3, 7)
    plt.imshow(clean_image, cmap='hot', origin='lower')
    plt.colorbar(label="Amplitude")
    plt.title("Cleaned Image")

    # plt.subplot(3, 3, 8)
    # plt.imshow(cnn_image, cmap='hot', origin='lower')
    # plt.colorbar(label="Amplitude")
    # plt.title("CNN Reconstructed Image")

    plt.tight_layout()
    plt.show()


hour_angle = 45
declination = -23.0229
wavelength = 3
image_size = 256

antenna_positions = parse_config_file('arrays/ALMA_cycle6_1.config')
baselines = compute_baselines(antenna_positions)

uv_coordinates = calculate_uv_coordinates(
    baselines, hour_angle, declination, wavelength)

sky_image = load_custom_sky_image('models/double_wide.png', image_size)
visibilities, model_fft = generate_visibilities(sky_image, uv_coordinates)
dirty_image, observed_fft = generate_dirty_image(
    visibilities, uv_coordinates, image_size)
beam_image = generate_beam_image(uv_coordinates, image_size)

clean_image, _, _, _ = clean(dirty_image, beam_image)


clean_images, dirty_images = generate_dataset([
    'models/galaxy_lobes.png',
    'models/galaxy_spiral.png',
    'models/blackhole_test.png',
    'models/rotated.png'
])

cnn_model = ImageReconstructionNet().to(device)
train_dataset = ImageDataset(dirty_images, clean_images)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
train_model(cnn_model, train_loader, num_epochs=20, learning_rate=3e-5)

cnn_model.eval()

with torch.no_grad():
    cnn_input = torch.tensor(
        dirty_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cnn_output = cnn_model(cnn_input.to(device)).squeeze().cpu().numpy()

    plot_all_results(sky_image, uv_coordinates, beam_image, model_fft,
                     observed_fft, dirty_image, clean_image, cnn_output)
