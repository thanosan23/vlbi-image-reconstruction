import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import ehtim as eh
from pyproj import Transformer
from ehtim import array, obsdata
from ehtim.observing import obs_simulate


transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)


def latlon_to_ecef(lat, lon, alt=0):
    """Convert latitude, longitude, and altitude to ECEF coordinates."""
    x, y, z = transformer.transform(lon, lat, alt)
    return x, y, z


def modify_telescope_positions(eht_array, new_positions):
    """Modify telescope positions in the EHT array."""
    for telescope_name, lat, lon, alt in new_positions:
        x, y, z = latlon_to_ecef(lat, lon, alt)
        idx = np.where(eht_array.tarr['site'] == telescope_name)[0]
        if len(idx) > 0:
            eht_array.tarr[idx[0]]['x'] = x
            eht_array.tarr[idx[0]]['y'] = y
            eht_array.tarr[idx[0]]['z'] = z
        else:
            print(f"Warning: Telescope '{
                  telescope_name}' not found in the array.")
    return eht_array


eht = eh.array.load_txt('../arrays/EHT2017.txt')
new_positions = []
eht = modify_telescope_positions(eht, new_positions)

image = np.array(Image.open(
    "../models/RIAF.png").resize((69, 69)).convert("L"))

eht_fov = 200 * eh.RADPERUAS
eht_size = image.shape[0]
im = eh.image.Image(
    image,
    psize=eht_fov / eht_size,
    ra=0.0,
    dec=0.0,
    rf=230e9,
    source='SyntheticImage'
)

obs = im.observe(
    eht, 60, 600, 0, 10, 4e9,
    sgrscat=False, ampcal=True, phasecal=True, ttype='fast', verbose=False
)

dim = obs.dirtyimage(image.shape[0], eht_fov).imarr()

vis = obs.data['vis']
real = (np.real(vis) - np.mean(np.real(vis))) / np.std(np.real(vis))
imag = (np.imag(vis) - np.mean(np.imag(vis))) / np.std(np.imag(vis))
u_coords = (obs.data['u'] - np.mean(obs.data['u'])) / np.std(obs.data['u'])
v_coords = (obs.data['v'] - np.mean(obs.data['v'])) / np.std(obs.data['v'])

kernel = (
    Matern(nu=1.5)
)

real_gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10)
imag_gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10)

baselines = np.vstack((u_coords, v_coords)).T
real_gp.fit(baselines, real)
imag_gp.fit(baselines, imag)

max_u = np.max(u_coords)
max_v = np.max(v_coords)
num_ghost_points = 100

ghost_u = np.random.uniform(max_u, 2 * max_u, num_ghost_points)
ghost_v = np.random.uniform(max_v, 2 * max_v, num_ghost_points)
ghost_real = np.zeros(num_ghost_points)
ghost_imag = np.zeros(num_ghost_points)

gu_coords = np.hstack([u_coords, ghost_u])
gv_coords = np.hstack([v_coords, ghost_v])
greal = np.hstack([real, ghost_real])
gimag = np.hstack([imag, ghost_imag])
grid_baselines = np.vstack((gu_coords, gv_coords)).T
real_gp.fit(grid_baselines, greal)
imag_gp.fit(grid_baselines, gimag)
grid_resolution = image.shape[0]

uv_points = obs_simulate.make_uvpoints(
    array=eht,
    ra=0.0,
    dec=0.0,
    rf=230e9,
    bw=4e9,
    tint=60,
    tadv=600,
    tstart=0,
    tstop=24,
)

u = []
v = []
for i in uv_points:
    u.append(i[6])
    v.append(i[7])

u = np.array(u)
v = np.array(v)

u_grid = (u -
          np.mean(obs.data['u'])) / np.std(obs.data['u'])
v_grid = (v -
          np.mean(obs.data['v'])) / np.std(obs.data['v'])

uu, vv = np.meshgrid(u_grid, v_grid)
grid_coords = np.vstack([uu.ravel(), vv.ravel()]).T
print(grid_coords.shape)

real_pred, _ = real_gp.predict(grid_coords, return_std=True)
imag_pred, _ = imag_gp.predict(grid_coords, return_std=True)

real_pred = real_pred * np.std(np.real(vis)) + np.mean(np.real(vis))
imag_pred = imag_pred * np.std(np.imag(vis)) + np.mean(np.imag(vis))
print("Done prediction!")

fft_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.complex128)
u_extent = gu_coords.max() - gu_coords.min()
v_extent = gv_coords.max() - gv_coords.min()

for coord, real_val, imag_val in zip(grid_coords, real_pred, imag_pred):
    u_idx = int((coord[0] - gu_coords.min()) /
                (gu_coords.max() - gu_coords.min()) * (grid_resolution - 1))
    v_idx = int((coord[1] - gv_coords.min()) /
                (gv_coords.max() - gv_coords.min()) * (grid_resolution - 1))

    if 0 <= u_idx < grid_resolution and 0 <= v_idx < grid_resolution:
        fft_grid[v_idx, u_idx] += real_val + 1j * imag_val

fft_grid = np.nan_to_num(fft_grid)

reconstructed_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_grid)))
reconstructed_image /= np.max(reconstructed_image)

original_resized = np.array(Image.fromarray(image).resize(
    reconstructed_image.shape, Image.BILINEAR))
original_resized = original_resized / np.max(original_resized)  # Normalize
ssim_score = ssim(original_resized, reconstructed_image,
                  data_range=reconstructed_image.max() - reconstructed_image.min())
mse_score = np.mean((original_resized - reconstructed_image)**2)

fig, ax = plt.subplots(2, 3, figsize=(13, 8))

ax[0, 0].imshow(image, cmap='hot')
ax[0, 0].set_title("Original Image")

ax[0, 1].imshow(dim, cmap='hot')
ax[0, 1].set_title("Dirty Image")

sc = ax[0, 2].scatter(u_coords, v_coords, c=np.abs(vis), cmap='viridis')
plt.colorbar(sc, ax=ax[0, 2], label='Visibility Amplitude')
ax[0, 2].set_title("UV Coverage")

sc_pred = ax[1, 0].scatter(grid_coords[:, 0], grid_coords[:, 1], c=np.abs(
    real_pred + 1j * imag_pred), cmap='plasma')
plt.colorbar(sc_pred, ax=ax[1, 0], label='Predicted Visibility Amplitude')
ax[1, 0].set_title("Predicted UV Coverage")

ax[1, 1].imshow(reconstructed_image, cmap='hot')
ax[1, 1].set_title("Reconstructed Image")

ax[1, 2].text(0.1, 0.5, f"SSIM: {ssim_score:.4f}\nMSE: {
              mse_score:.4f}", fontsize=12)
ax[1, 2].axis('off')

plt.tight_layout()
plt.show()
