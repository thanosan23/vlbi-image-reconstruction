import tkinter as tk
from tkinter import filedialog
import numpy as np
from itertools import combinations
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter


class TelescopeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Telescope Array Simulator")
        self.image_size = 256
        self.wavelength = 3
        self.hour_angle = 45
        self.declination = -23.0229

        self.antenna_positions = self.parse_config_file(
            "arrays/ALMA_cycle6_1.config")

        self.dragged_marker = None
        self.dragged_index = None

        self.setup_ui()

        self.sky_image_path = "models/double_wide.png"
        self.load_sky_image(self.sky_image_path)
        self.update_results()

    def parse_config_file(self, file_path):
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

    def compute_baselines(self, antenna_positions):
        baselines = []
        for (i, pos1), (j, pos2) in combinations(enumerate(antenna_positions),
                                                 2):
            baseline = pos2 - pos1
            baselines.append(baseline)
        return np.array(baselines)

    def calculate_uv_coordinates(self, baselines):
        h0 = np.radians(self.hour_angle)
        delta0 = np.radians(self.declination)

        rotation_matrix = np.array([
            [np.sin(h0), np.cos(h0), 0],
            [-np.sin(delta0) * np.cos(h0), np.sin(delta0)
             * np.sin(h0), np.cos(delta0)],
            [np.cos(delta0) * np.cos(h0), -np.cos(delta0)
             * np.sin(h0), np.sin(delta0)],
        ])

        baselines_3d = np.hstack(
            (baselines, np.zeros((baselines.shape[0], 1))))
        uvw_coordinates = np.dot(
            baselines_3d, rotation_matrix.T) / self.wavelength
        return uvw_coordinates[:, :2]

    def load_sky_image(self, image_path):
        img = Image.open(image_path).convert("L")
        img = img.resize((self.image_size, self.image_size))
        self.sky_image = np.array(img) / 255.0

    def generate_dirty_image(self, uv_coordinates):
        ft_image = fftshift(fft2(ifftshift(self.sky_image)))
        visibilities = []
        image_center = np.array(self.sky_image.shape) // 2
        for u, v in uv_coordinates:
            u = int(round(u)) + image_center[0]
            v = int(round(v)) + image_center[1]
            if 0 <= u < self.sky_image.shape[0] and 0 <= v < self.sky_image.shape[1]:
                visibilities.append(ft_image[u, v])
            else:
                visibilities.append(0)
        visibilities = np.array(visibilities)

        ft_dirty_image = np.zeros(
            (self.image_size, self.image_size), dtype=complex)
        for (u, v), vis in zip(uv_coordinates, visibilities):
            u_idx = int(round(u)) + image_center[0]
            v_idx = int(round(v)) + image_center[1]
            if 0 <= u_idx < self.image_size and 0 <= v_idx < self.image_size:
                ft_dirty_image[u_idx, v_idx] = vis

        dirty_image = np.abs(ifftshift(ifft2(fftshift(ft_dirty_image))))
        return dirty_image

    def gaussian_clean_beam(self, beam_image):
        return gaussian_filter(beam_image, sigma=5)

    def clean(self, dirty_image, beam_image, gamma=0.1, threshold=0.001,
              max_iterations=100):
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

        clean_beam = self.gaussian_clean_beam(beam_image)
        clean_image = np.abs(
            ifftshift(ifft2(
                fftshift(fft2(clean_components) * fft2(clean_beam))
            )))
        return clean_image, clean_components, residuals, clean_beam

    def update_results(self):
        baselines = self.compute_baselines(self.antenna_positions)
        uv_coordinates = self.calculate_uv_coordinates(baselines)

        self.dirty_image = self.generate_dirty_image(uv_coordinates)

        self.beam_image = self.generate_beam_image(uv_coordinates)

        self.clean_image, _, _, _ = self.clean(
            self.dirty_image, self.beam_image)

        self.model_fft = fft2(self.sky_image)
        self.observed_fft = fft2(self.dirty_image)

        self.update_plots(uv_coordinates)

    def generate_beam_image(self, uv_coordinates):
        ft_beam_image = np.zeros(
            (self.image_size, self.image_size), dtype=complex)
        image_center = np.array((self.image_size // 2, self.image_size // 2))
        for u, v in uv_coordinates:
            u_idx = int(round(u)) + image_center[0]
            v_idx = int(round(v)) + image_center[1]
            if 0 <= u_idx < self.image_size and 0 <= v_idx < self.image_size:
                ft_beam_image[u_idx, v_idx] = 1
                ft_beam_image[-u_idx % self.image_size, -v_idx %
                              self.image_size] = 1

        beam_image = np.abs(ifftshift(ifft2(fftshift(ft_beam_image))))
        return beam_image

    def update_plots(self, uv_coordinates):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        self.ax7.clear()
        self.ax8.clear()

        self.ax1.imshow(self.sky_image, cmap='hot', origin='lower')
        self.ax1.set_title("Model Image (Sky)")

        self.ax2.imshow(self.beam_image, cmap='hot', origin='lower')
        self.ax2.set_title("Beam Image (PSF)")

        if len(uv_coordinates) > 0:
            self.ax3.plot(uv_coordinates[:, 0],
                          uv_coordinates[:, 1], 'bo', markersize=2)
            self.ax3.plot(-uv_coordinates[:, 0], -
                          uv_coordinates[:, 1], 'ro', markersize=2)
        self.ax3.set_xlabel("u (wavelengths)")
        self.ax3.set_ylabel("v (wavelengths)")
        self.ax3.set_title("UV Coverage")
        self.ax3.grid(True)
        self.ax3.axis("equal")

        self.ax4.imshow(np.log(np.abs(self.model_fft)),
                        cmap='hot', origin='lower')
        self.ax4.set_title("Model FFT")

        self.ax5.imshow(np.log(np.abs(self.observed_fft) +
                        1e-10), cmap='hot', origin='lower')
        self.ax5.set_title("Observed FFT")

        self.ax6.imshow(self.dirty_image, cmap='hot', origin='lower')
        self.ax6.set_title("Dirty Image")

        self.ax7.imshow(self.clean_image, cmap='hot', origin='lower')
        self.ax7.set_title("Cleaned Image")

        self.ax8.set_title("Telescope Array")
        self.ax8.set_xlabel("E-W Position")
        self.ax8.set_ylabel("N-S Position")
        self.ax8.grid(True)

        self.add_telescope_markers()

        self.canvas.draw()

    def add_telescope_markers(self):
        for i, (e_offset, n_offset) in enumerate(self.antenna_positions):
            marker = self.ax8.plot(
                e_offset, n_offset, 'go', markersize=8, picker=5)
            marker[0].set_gid(i)

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=20, pady=20)

        tk.Label(control_frame, text="Hour Angle").pack()
        self.hour_angle_slider = tk.Scale(
            control_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
            command=self.set_hour_angle)
        self.hour_angle_slider.set(self.hour_angle)
        self.hour_angle_slider.pack()

        tk.Label(control_frame, text="Declination").pack()
        self.declination_slider = tk.Scale(
            control_frame, from_=-90, to=90, orient=tk.HORIZONTAL,
            command=self.set_declination)
        self.declination_slider.set(self.declination)
        self.declination_slider.pack()

        tk.Label(control_frame, text="Wavelength").pack()
        self.wavelength_slider = tk.Scale(
            control_frame, from_=1, to=10, orient=tk.HORIZONTAL,
            command=self.set_wavelength)
        self.wavelength_slider.set(self.wavelength)
        self.wavelength_slider.pack()

        tk.Button(control_frame, text="Load Sky Image",
                  command=self.load_new_image).pack(pady=10)

        self.fig = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.ax8 = self.fig.add_subplot(3, 3, 1)
        self.ax1 = self.fig.add_subplot(3, 3, 2)
        self.ax2 = self.fig.add_subplot(3, 3, 3)
        self.ax3 = self.fig.add_subplot(3, 3, 4)
        self.ax4 = self.fig.add_subplot(3, 3, 5)
        self.ax5 = self.fig.add_subplot(3, 3, 6)
        self.ax6 = self.fig.add_subplot(3, 3, 7)
        self.ax7 = self.fig.add_subplot(3, 3, 8)

        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig.canvas.mpl_connect('pick_event', self.on_marker_pick)

    def on_marker_pick(self, event):
        if isinstance(event.artist, mlines.Line2D):
            self.dragged_marker = event.artist
            self.dragged_index = event.artist.get_gid()
            self.fig.canvas.mpl_connect(
                'motion_notify_event', self.on_mouse_move)
            self.fig.canvas.mpl_connect(
                'button_release_event', self.on_mouse_release)

    def on_mouse_move(self, event):
        if self.dragged_marker is not None:
            x, y = event.xdata, event.ydata
            self.dragged_marker.set_data(x, y)
            self.antenna_positions[self.dragged_index] = [x, y]
            self.update_results()
            self.canvas.draw()

    def on_mouse_release(self, event):
        if self.dragged_marker is not None:
            self.dragged_marker = None
            self.fig.canvas.mpl_disconnect('motion_notify_event')
            self.fig.canvas.mpl_disconnect('button_release_event')

    def load_new_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.sky_image_path = file_path
            self.load_sky_image(file_path)
            self.update_results()

    def set_hour_angle(self, val):
        self.hour_angle = float(val)
        self.update_results()

    def set_declination(self, val):
        self.declination = float(val)
        self.update_results()

    def set_wavelength(self, val):
        self.wavelength = float(val)
        self.update_results()


if __name__ == "__main__":
    root = tk.Tk()
    app = TelescopeApp(root)
    root.mainloop()
