import sys
import os
from astropy.time import Time
import traceback
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter
from mpl_toolkits.basemap import Basemap
import datetime
from astropy.io import fits
import torch
import torch.nn as nn
from torchvision import transforms
import ehtim as eh
import ehtim as eh
from PIL import Image
import numpy as np
from pyproj import Proj, Transformer
from telescope import TelescopeArray
from config import TelescopeConfig

geodetic = Proj(proj='latlong', datum='WGS84')
ecef = Proj(proj='latlong', datum='WGS84',
            lat_0=0, lon_0=0, x_0=0, y_0=0, z_0=0)

transformer = Transformer.from_proj(geodetic, ecef)
transformer2 = Transformer.from_proj(ecef, geodetic)


def latlon_to_ecef(lat, lon, alt=0):
    x, y, z = transformer.transform(lon, lat, alt)
    return x, y, z

def ecef_to_latlon(x, y, z):
    lon, lat, alt = transformer2.transform(x, y, z, radians=False)
    return lat, lon


def modify_telescope_positions(eht, new_positions):
    for telescope_name, lat, lon, alt in new_positions:
        x, y, z = latlon_to_ecef(lat, lon, alt)

        idx = np.where(eht.tarr['site'] == telescope_name)[0]
        if len(idx) > 0:
            eht.tarr[idx[0]]['x'] = x
            eht.tarr[idx[0]]['y'] = y
            eht.tarr[idx[0]]['z'] = z
        else:
            print(f"Warning: Telescope '{
                  telescope_name}' not found in the array.")
    return eht


def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2 * normalized_input - 1


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


class TelescopeApp:
    def __init__(self, root):
        self.root = root
        self.current_mode = "VLA"

        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)

        self.image_size = 256
        self.wavelength = 1.33
        self.hour_angle = 45
        self.declination = 34.078745
        self.clean_gamma = 0.1
        self.clean_threshold = 0.1
        self.start_time = datetime.datetime.now()

        self.eht = None

        self.eht = eh.array.load_txt('arrays/EHT2017.txt')

        self.new_positions = []
        self.eht = modify_telescope_positions(self.eht, self.new_positions)

        self.dirty_image = None
        self.beam_image = None

        self.selected_telescope = None

        self.mode_var = tk.StringVar(value="VLA")
        self.wavelength_var = tk.DoubleVar(value=self.wavelength)
        self.hour_angle_var = tk.DoubleVar(value=self.hour_angle)
        self.declination_var = tk.DoubleVar(value=self.declination)
        self.clean_gamma_var = tk.DoubleVar(value=self.clean_gamma)
        self.clean_threshold_var = tk.DoubleVar(value=self.clean_threshold)
        self.time_var = tk.StringVar(
            value=self.start_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.duration_var = tk.DoubleVar(value=24.0)

        self.edit_source_position_var = tk.BooleanVar(value=False)

        self.telescope_array = TelescopeArray(mode="VLA")

        self.telescope_list = None

        control_panel = ttk.Frame(main_container)
        control_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.setup_control_sections(control_panel)
        self.setup_telescope_management(control_panel)

        if self.current_mode == "VLA":
            self.satellite_frame.pack_forget()

        plot_panel = ttk.Frame(main_container)
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_plots(plot_panel)

        self.model = UNetModel(in_channels=1, out_channels=1, channels=64,
                               num_pool_layers=4, drop_prob=0.2)
        self.model.load_state_dict(torch.load(
            "./unet/unet_galaxy10.pth", weights_only=True))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.eht_fov = 200 * eh.RADPERUAS

        self.file_path = "models/double_wide.png"
        self.load_sky_image("models/double_wide.png")

        self.ax_unet = None

        print(self.eht.tarr)
        self.update_results()

    def setup_telescope_management(self, parent):
        self.update_telescope_list()

    def update_telescope_list(self):
        self.telescope_listbox.delete(0, tk.END)

        if self.mode_var.get() == "VLA":
            for i, pos in enumerate(self.telescope_array.positions):
                self.telescope_listbox.insert(
                    tk.END, f"Telescope {i}: E={pos[0]:.2f}, N={pos[1]:.2f}"
                )
        else:
            for i, (name, (lat, lon)) in enumerate(zip(self.telescope_array.names,
                                                       self.telescope_array.lat_lon)):
                self.telescope_listbox.insert(
                    tk.END, f"{name}: Lat={lat:.2f}, Lon={lon:.2f}"
                )

        self.root.update_idletasks()

    def add_telescope(self):
        if self.mode_var.get() == "VLA":
            file_path = filedialog.askopenfilename(
                initialdir="arrays",
                title="Select Array Configuration",
                filetypes=[("Config files", "*.config")]
            )
            self.file_path = file_path
            if file_path:
                new_positions = self.telescope_array.parse_config_file(
                    file_path)
                if new_positions is not None:
                    if self.telescope_array.positions is None:
                        self.telescope_array.positions = new_positions
                    else:
                        self.telescope_array.positions = np.vstack([
                            self.telescope_array.positions,
                            new_positions
                        ])
                    self.telescope_array.compute_baselines()
                    self.update_telescope_list()
                    self.update_results()
        else:
            new_name = f"Tel{len(self.telescope_array.names) + 1}"
            new_lat_lon = (0.0, 0.0)
            self.telescope_array.names.append(new_name)
            if self.telescope_array.lat_lon is None:
                self.telescope_array.lat_lon = np.array([new_lat_lon])
            else:
                self.telescope_array.lat_lon = np.vstack(
                    [self.telescope_array.lat_lon, new_lat_lon])
            self.new_positions.append(
                (new_name, new_lat_lon[0], new_lat_lon[1], 0.0))

            if self.add_satellite_var.get():
                period_days = self.period_days_var.get()
                eccentricity = self.eccentricity_var.get()
                inclination = self.inclination_var.get()
                arg_perigee = self.arg_perigee_var.get()

                self.eht = self.eht.add_satellite_elements(
                    new_name,
                    period_days=period_days,
                    eccentricity=eccentricity,
                    inclination=inclination,
                    arg_perigee=arg_perigee,
                )
            else:
                self.eht = self.eht.add_site(
                    new_name, (new_lat_lon[0], new_lat_lon[1], 0.0))

            print(self.eht.tarr)
            self.telescope_array.compute_baselines()
            self.update_telescope_list()
            self.update_results()

    def remove_telescope(self):
        self.telescope_listbox.focus_set()
        selection = self.telescope_listbox.curselection()
        if not selection:
            return

        print(selection, self.telescope_array.names)
        idx = selection[0]

        if self.mode_var.get() == "VLA":
            if len(self.telescope_array.positions) > 1:
                self.telescope_array.positions = np.delete(
                    self.telescope_array.positions, idx, axis=0)
                self.telescope_array.compute_baselines()
            else:
                print("Cannot remove the last VLA telescope")
        else:
            if len(self.telescope_array.names) > 2:
                telescope_name = self.telescope_array.names[idx]
                if idx == len(self.telescope_array.names) - 1:
                    self.telescope_array.names.pop()
                else:
                    self.telescope_array.names.pop(idx)
                self.telescope_array.lat_lon = np.delete(
                    self.telescope_array.lat_lon, idx, axis=0)

                self.new_positions = [
                    pos for pos in self.new_positions if pos[0] != telescope_name]

                self.telescope_array.compute_baselines()

                self.eht = self.eht.remove_site(telescope_name)
                print(self.eht.tarr)
            else:
                print("Cannot remove the last EHT telescope")

        self.update_telescope_list()
        self.update_results()
    
    def setup_control_sections(self, parent):
        style = ttk.Style()
        style.configure('Modern.TLabelframe', padding=10)
        style.configure('Modern.TButton', padding=5)
        style.configure('Horizontal.TScale', background='#f0f0f0')

        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(expand=True, fill=tk.BOTH)
        
        x_scrollbar = ttk.Scrollbar(control_frame, orient="horizontal")
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        y_scrollbar = ttk.Scrollbar(control_frame, orient="vertical")
        y_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        control_canvas = tk.Canvas(control_frame, yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        control_canvas.pack(fill=tk.BOTH, expand=True)

        y_scrollbar.config(command=control_canvas.yview)
        x_scrollbar.config(command=control_canvas.xview)

        scrollable_frame = ttk.Frame(control_canvas, padding=10)
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )

        mode_frame = ttk.LabelFrame(scrollable_frame, text="Array Mode", style='Modern.TLabelframe')
        mode_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        self.mode_var = tk.StringVar(value=self.current_mode)
        ttk.Radiobutton(mode_frame, text="VLA", variable=self.mode_var, value="VLA",
                        command=self.change_mode).pack(side=tk.LEFT, padx=20, pady=5)
        ttk.Radiobutton(mode_frame, text="EHT", variable=self.mode_var, value="EHT",
                        command=self.change_mode).pack(side=tk.LEFT, padx=20, pady=5)

        array_frame = ttk.LabelFrame(scrollable_frame, text="Array Parameters", style='Modern.TLabelframe')
        array_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        def create_slider_with_label(frame, label_text, variable, from_, to):
            slider_frame = ttk.Frame(frame)
            slider_frame.pack(fill=tk.X, pady=5)
            ttk.Label(slider_frame, text=label_text).pack(side=tk.LEFT)
            value_label = ttk.Label(slider_frame, text=f"{variable.get():.2f}")
            value_label.pack(side=tk.RIGHT, padx=5)
            slider = ttk.Scale(slider_frame, from_=from_, to=to, variable=variable,
                            orient=tk.HORIZONTAL, command=lambda v: value_label.config(text=f"{float(v):.2f}"))
            slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        create_slider_with_label(array_frame, "Wavelength (mm):", self.wavelength_var, 0.01, 5)
        create_slider_with_label(array_frame, "Duration (hrs):", self.duration_var, 1, 96)

        clean_frame = ttk.LabelFrame(scrollable_frame, text="CLEAN Parameters", style='Modern.TLabelframe')
        clean_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        create_slider_with_label(clean_frame, "Loop Gain:", self.clean_gamma_var, 0.01, 1.0)
        create_slider_with_label(clean_frame, "Threshold:", self.clean_threshold_var, 0.0001, 1)

        telescope_management = ttk.LabelFrame(scrollable_frame, text="Telescope Management", style='Modern.TLabelframe')
        telescope_management.pack(fill=tk.X, pady=10, padx=5)
        self.telescope_listbox = tk.Listbox(telescope_management, height=10, width=30)
        self.telescope_listbox.pack(pady=5, padx=3)

        ttk.Button(telescope_management, text="Add Telescope", command=self.add_telescope).pack(side=tk.LEFT, pady=5, padx=3)
        ttk.Button(telescope_management, text="Remove Selected", command=self.remove_telescope).pack(side=tk.LEFT, pady=5, padx=3)

        ttk.Checkbutton(telescope_management, text="Edit Source Position", variable=self.edit_source_position_var).pack(side=tk.LEFT, padx=5, pady=5)

        model_frame = ttk.Frame(scrollable_frame)
        model_frame.pack(fill=tk.X, pady=10, padx=5)
        ttk.Button(model_frame, text="Load Sky Image", command=self.load_new_image,
                style='Modern.TButton').pack(fill=tk.X, pady=5)

        update_frame = ttk.Frame(scrollable_frame)
        update_frame.pack(fill=tk.X, pady=10, padx=5)
        self.update_button = ttk.Button(update_frame, text="Update Results", command=self.update_results, style='Modern.TButton')
        self.update_button.pack(fill=tk.X, pady=5)

        self.satellite_frame = ttk.LabelFrame(scrollable_frame, text="Satellite Parameters", style='Modern.TLabelframe')
        self.satellite_frame.pack(fill=tk.X, pady=10)

        self.add_satellite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.satellite_frame, text="Add Satellite", variable=self.add_satellite_var).pack(side=tk.LEFT, padx=5, pady=5)

        self.period_days_var = tk.DoubleVar(value=1.0)
        self.eccentricity_var = tk.DoubleVar(value=0.0)
        self.inclination_var = tk.DoubleVar(value=0.0)
        self.arg_perigee_var = tk.DoubleVar(value=0.0)
        self.long_ascending_var = tk.DoubleVar(value=0.0)

        create_slider_with_label(self.satellite_frame, "Period (days):", self.period_days_var, 0.1, 10)
        create_slider_with_label(self.satellite_frame, "Eccentricity:", self.eccentricity_var, 0.0, 1.0)
        create_slider_with_label(self.satellite_frame, "Inclination:", self.inclination_var, 0.0, 180.0)
        create_slider_with_label(self.satellite_frame, "Arg Perigee:", self.arg_perigee_var, 0.0, 180.0)
        create_slider_with_label(self.satellite_frame, "Long Ascending:", self.long_ascending_var, 0.0, 360.0)

    def set_time(self):
        current_time = datetime.datetime.strptime(
            self.time_var.get(), "%Y-%m-%d %H:%M:%S")

        dialog = tk.Toplevel(self.root)
        dialog.title("Set Start Time")
        dialog.geometry("300x200")

        try:
            from tkcalendar import DateEntry
            cal = DateEntry(dialog, width=12, background='darkblue',
                            foreground='white', borderwidth=2)
            cal.pack(pady=10)
        except ImportError:
            cal = ttk.Entry(dialog)
            cal.insert(0, current_time.strftime("%Y-%m-%d"))
            cal.pack(pady=10)

        time_frame = ttk.Frame(dialog)
        time_frame.pack(pady=10)
        hour_var = tk.StringVar(value=current_time.strftime("%H"))
        min_var = tk.StringVar(value=current_time.strftime("%M"))

        ttk.Entry(time_frame, textvariable=hour_var,
                  width=3).pack(side=tk.LEFT)
        ttk.Label(time_frame, text=":").pack(side=tk.LEFT)
        ttk.Entry(time_frame, textvariable=min_var, width=3).pack(side=tk.LEFT)

        def apply():
            try:
                date_str = cal.get() if hasattr(cal, 'get') else cal.get()
                time_str = f"{hour_var.get()}:{min_var.get()}:00"
                new_time = datetime.datetime.strptime(f"{date_str} {time_str}",
                                                      "%Y-%m-%d %H:%M:%S")
                self.time_var.set(new_time.strftime("%Y-%m-%d %H:%M:%S"))
                dialog.destroy()
                self.update_results()
            except ValueError as e:
                tk.messagebox.showerror("Error", f"Invalid time format: {e}")

        ttk.Button(dialog, text="Apply", command=apply).pack(pady=10)

    def setup_plots(self, parent):
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()

        if self.mode_var.get() == "EHT":
            self.ax_map = self.fig.add_subplot(331)
            self.ax_model = self.fig.add_subplot(332)
            self.ax_beam = self.fig.add_subplot(333)
            self.ax_uv = self.fig.add_subplot(334)
            self.ax_model_fft = self.fig.add_subplot(335)
            self.ax_dirty = self.fig.add_subplot(336)
            self.ax_clean = self.fig.add_subplot(337)
            self.ax_unet = self.fig.add_subplot(338)
            self.plot_earth_map()
        else:
            self.ax_array = self.fig.add_subplot(331)
            self.ax_model = self.fig.add_subplot(332)
            self.ax_beam = self.fig.add_subplot(333)
            self.ax_uv = self.fig.add_subplot(334)
            self.ax_model_fft = self.fig.add_subplot(335)
            self.ax_dirty = self.fig.add_subplot(336)
            self.ax_clean = self.fig.add_subplot(337)

        self.fig.tight_layout()

        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH,
                                         expand=True)
        self.setup_telescope_management(self.canvas.get_tk_widget().master)

    def change_mode(self):
        mode = self.mode_var.get()

        if mode == "EHT":
            self.satellite_frame.pack()
        else:
            self.satellite_frame.pack_forget()

        self.telescope_array = TelescopeArray(mode=mode)

        for widget in self.canvas.get_tk_widget().master.winfo_children():
            widget.destroy()

        self.setup_plots(self.canvas.get_tk_widget().master)

        self.update_mode_controls()
        self.update_telescope_list()
        self.update_results()

    def update_mode_controls(self):
        mode = self.mode_var.get()
        return mode

    def load_sky_image(self, image_path):
        self.eht_image = self.load_image(image_path)

        if image_path.endswith(".fits"):
            img_data = fits.getdata(image_path)

            img_data = img_data - np.min(img_data)
            img_data = img_data / np.max(img_data) * 255.0

            img = Image.fromarray(img_data.astype(np.uint8), 'L')
        else:
            img = Image.open(image_path).convert("L")

        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        self.sky_image = np.array(img) / 255.0

    def generate_dirty_image(self, uv_coordinates):
        ft_image = fftshift(fft2(ifftshift(self.sky_image)))
        visibilities = []
        image_center = np.array(self.sky_image.shape) // 2

        max_uv = np.max(np.abs(uv_coordinates))
        scale_factor = (self.image_size // 2) / max_uv

        for u, v in uv_coordinates:
            u_scaled = int(round(u * scale_factor)) + image_center[0]
            v_scaled = int(round(v * scale_factor)) + image_center[1]
            if 0 <= u_scaled < self.image_size and 0 <= v_scaled < self.image_size:
                visibilities.append(ft_image[u_scaled, v_scaled])
            else:
                visibilities.append(0)

        visibilities = np.array(visibilities)

        ft_dirty_image = np.zeros(
            (self.image_size, self.image_size), dtype=complex)

        for (u, v), vis in zip(uv_coordinates, visibilities):
            u_scaled = int(round(u * scale_factor)) + image_center[0]
            v_scaled = int(round(v * scale_factor)) + image_center[1]
            if 0 <= u_scaled < self.image_size and 0 <= v_scaled < self.image_size:
                ft_dirty_image[u_scaled, v_scaled] += vis
                ft_dirty_image[-u_scaled % self.image_size, -
                               v_scaled % self.image_size] += np.conj(vis)

        dirty_image = np.abs(ifftshift(ifft2(fftshift(ft_dirty_image))))
        return dirty_image

    def generate_beam_image(self, uv_coordinates):
        ft_beam_image = np.zeros(
            (self.image_size, self.image_size), dtype=complex)
        image_center = np.array((self.image_size // 2, self.image_size // 2))

        max_uv = np.max(np.abs(uv_coordinates))
        scale_factor = (self.image_size // 4) / max_uv

        for u, v in uv_coordinates:
            u_scaled = int(round(u * scale_factor)) + image_center[0]
            v_scaled = int(round(v * scale_factor)) + image_center[1]

            if (0 <= u_scaled < self.image_size and
                    0 <= v_scaled < self.image_size):
                ft_beam_image[u_scaled, v_scaled] = 1
                ft_beam_image[-u_scaled %
                              self.image_size, -v_scaled % self.image_size] = 1

        beam_image = np.abs(ifftshift(ifft2(fftshift(ft_beam_image))))
        beam_image /= np.max(beam_image)
        return beam_image

    def generate_uv_coverage(self):
        uv_coordinates = []

        if self.mode_var.get() == "VLA":
            for i in range(len(self.telescope_array.positions)):
                for j in range(i + 1, len(self.telescope_array.positions)):
                    u = self.telescope_array.positions[j][0] - \
                        self.telescope_array.positions[i][0]
                    v = self.telescope_array.positions[j][1] - \
                        self.telescope_array.positions[i][1]
                    uv_coordinates.append((u, v))
                    uv_coordinates.append((-u, -v))

        elif self.mode_var.get() == "EHT":
            print(self.eht.data['u'])
            print(self.eht.data['v'])
            for i in range(len(self.telescope_array.lat_lon)):
                for j in range(i + 1, len(self.telescope_array.lat_lon)):
                    lat1, lon1 = self.telescope_array.lat_lon[i]
                    lat2, lon2 = self.telescope_array.lat_lon[j]

                    u = lon2 - lon1
                    v = lat2 - lat1
                    uv_coordinates.append((u, v))
                    uv_coordinates.append((-u, -v))
        print(uv_coordinates)
        return np.array(uv_coordinates)

    def clean(self, dirty_image, beam_image, max_iterations=100):
        cleaned = np.zeros_like(dirty_image)
        residual = dirty_image.copy()

        beam_max = np.max(beam_image)
        beam_image = beam_image / beam_max
        beam_center = np.array(beam_image.shape) // 2

        components = []

        for iteration in range(max_iterations):
            max_pos = np.unravel_index(
                np.argmax(np.abs(residual)), residual.shape)
            max_val = residual[max_pos]

            if np.abs(max_val) < self.clean_threshold:
                break

            components.append((max_pos, max_val * self.clean_gamma))

            y_start = max_pos[0] - beam_center[0]
            y_end = y_start + beam_image.shape[0]
            x_start = max_pos[1] - beam_center[1]
            x_end = x_start + beam_image.shape[1]

            y_slice = slice(max(0, y_start), min(residual.shape[0], y_end))
            x_slice = slice(max(0, x_start), min(residual.shape[1], x_end))
            beam_y_slice = slice(
                max(0, -y_start), min(beam_image.shape[0], residual.shape[0] - y_start))
            beam_x_slice = slice(
                max(0, -x_start), min(beam_image.shape[1], residual.shape[1] - x_start))

            residual[y_slice, x_slice] -= max_val * self.clean_gamma * \
                beam_image[beam_y_slice, beam_x_slice]

        clean_beam = gaussian_filter(beam_image, sigma=0.001)
        clean_beam /= np.max(clean_beam)

        for (pos, amp) in components:
            y_start = pos[0] - beam_center[0]
            y_end = y_start + clean_beam.shape[0]
            x_start = pos[1] - beam_center[1]
            x_end = x_start + clean_beam.shape[1]

            y_slice = slice(max(0, y_start), min(cleaned.shape[0], y_end))
            x_slice = slice(max(0, x_start), min(cleaned.shape[1], x_end))
            beam_y_slice = slice(
                max(0, -y_start), min(clean_beam.shape[0], cleaned.shape[0] - y_start))
            beam_x_slice = slice(
                max(0, -x_start), min(clean_beam.shape[1], cleaned.shape[1] - x_start))

            cleaned[y_slice, x_slice] += amp * \
                clean_beam[beam_y_slice, beam_x_slice]

        return cleaned, components, residual, clean_beam

    def plot_earth_map(self):
        if not hasattr(self, 'ax_map'):
            return

        self.ax_map.clear()
        m = Basemap(projection='cyl', resolution='c',
                    llcrnrlat=-90, urcrnrlat=90,
                    llcrnrlon=-180, urcrnrlon=180,
                    ax=self.ax_map)

        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawmapboundary(fill_color='lightblue')

        for name, (lat, lon) in zip(self.telescope_array.names,
                                    self.telescope_array.lat_lon):
            x, y = m(lon, lat)
            print(lat, lon)
            m.plot(x, y, 'ro', markersize=6)
            self.ax_map.text(x + 2, y + 2, name, fontsize=8)

        for i, (lat1, lon1) in enumerate(self.telescope_array.lat_lon):
            for lat2, lon2 in self.telescope_array.lat_lon[i+1:]:
                x1, y1 = m(lon1, lat1)
                x2, y2 = m(lon2, lat2)
                self.ax_map.plot([x1, x2], [y1, y2], 'b-',
                                 alpha=0.3, linewidth=0.5)

        dec = self.declination_var.get()
        if -90 <= dec <= 90:
            lons = np.linspace(-180, 180, 360)
            decs = np.ones_like(lons) * dec
            x, y = m(lons, decs)
            m.plot(x, y, 'g--', alpha=0.5, label='Source declination')

            if hasattr(self, 'source_lon'):
                x, y = m(self.source_lon, dec)
                m.plot(x, y, 'g*', markersize=10, label='Source')

        self.ax_map.set_title(
            "EHT Telescope Locations\nClick to set source position")
        self.ax_map.grid(True)

        m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])

    def load_image(self, image_path):

        if image_path.lower().endswith('.fits'):
            im = eh.image.load_image(image_path)
        elif image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(image_path).convert('L')
            img = img.resize((69, 69), Image.LANCZOS)
            img_array = np.array(img)

            normalized_img = img_array / 255.0
            total_flux = 1.0
            eht_size = img_array.shape[0]

            im_array = normalized_img * total_flux / normalized_img.sum()

            freq = ((2.998*10e8)/(self.wavelength_var.get() * 10e-3))

            im = eh.image.Image(
                im_array,
                psize=self.eht_fov / eht_size,
                ra=self.hour_angle_var.get(),
                dec=self.declination_var.get(),
                rf=freq,
                source='SyntheticImage'
            )
        else:
            raise ValueError(f"Unsupported file format: {image_path}")

        return im

    def generate_eht_image(self, im, eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, npix=69,
                           ttype='direct', mjd=None):

        obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                         sgrscat=False, ampcal=True, phasecal=True, ttype=ttype, mjd=mjd,  no_elevcut_space=True,  dcal=True, add_th_noise=False)
        fov = 200 * eh.RADPERUAS

        dim = obs.dirtyimage(npix, fov)
        dbeam = obs.dirtybeam(npix, fov)
        cbeam = obs.cleanbeam(npix, fov)

        dim_array = dim.imarr()
        dbeam_array = dbeam.imarr()
        cbeam_array = cbeam.imarr()

        zbl = im.total_flux()

        prior_fwhm = 100 * eh.RADPERUAS
        emptyprior = eh.image.make_square(obs, npix, fov)
        gaussprior = emptyprior.add_gauss(
            zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

        data_term = {'vis': 1}
        reg_term = {'tv2': 1, 'l1': 0.1}

        imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, flux=zbl,
                                data_term=data_term, reg_term=reg_term,
                                norm_reg=True,
                                epsilon_tv=1.e-10,
                                maxit=250, ttype=ttype)
        imgr.make_image_I(show_updates=False)

        out = imgr.out_last()
        out = out.imarr()

        return out, dim_array, dbeam_array, cbeam_array

    def update_results(self):
        self.update_parameters()
        try:
            if self.mode_var.get() == "VLA":
                uv_coordinates = self.telescope_array.calculate_uv_coordinates(
                    self.hour_angle_var.get(),
                    self.declination_var.get()
                )
            else:
                try:
                    start_time = datetime.datetime.strptime(
                        self.time_var.get(), "%Y-%m-%d %H:%M:%S")
                    uv_coordinates = []
                    tint_sec = 60
                    tadv_sec = 600
                    tstart_hr = 0
                    tstop_hr = self.duration_var.get()
                    bw_hz = 4.e9
                    freq = ((2.998*10e8)/(self.wavelength_var.get()))
                    print("Analyzing at frequency: ", freq)
                    data = self.eht.obsdata(-self.hour_angle_var.get(), self.declination_var.get(
                    ), freq, bw_hz, tint_sec, tadv_sec, tstart_hr, tstop_hr)
                    u = []
                    v = []
                    for i in data.data:
                        u.append(i[6])
                        v.append(i[7])
                    u = np.array(u)
                    v = np.array(v)

                    uv_coordinates = np.column_stack((u, v))
                except ValueError:
                    print("Invalid time format")
                    return

            self.telescope_array.wavelength = self.wavelength_var.get() / 1000

            if self.mode_var.get() == "EHT":
                if self.eht is not None:
                    mjd_now = Time.now().mjd
                    self.eht_image = self.load_image(self.file_path)
                    self.eht_image.mjd = mjd_now
                    out, dim, dbeam, cbeam = self.generate_eht_image(
                        self.eht_image, self.eht, tint_sec, tadv_sec, tstart_hr, self.duration_var.get(), bw_hz, mjd=mjd_now)
                    self.dirty_image = dim
                    self.beam_image = dbeam
                    self.clean_image = out

                    self.dirty_image = normalize_negative_one(self.dirty_image)
                    test_image = Image.fromarray(self.dirty_image)
                    test_image = test_image.transpose(Image.FLIP_LEFT_RIGHT)
                    test_image = self.transform(test_image)
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(test_image.reshape(
                            1, 69, 69).unsqueeze(0)).squeeze(0).squeeze(0).numpy()
                        pred = normalize_negative_one(pred)
                        self.ax_unet.clear()
                        self.ax_unet.imshow(pred, cmap='hot', extent=[
                            -self.eht_fov / 2, self.eht_fov / 2, -self.eht_fov / 2, self.eht_fov / 2])
                        self.ax_unet.set_title("UNet Output")
                        self.ax_unet.set_xlabel('x (µas)')
                        self.ax_unet.set_ylabel('y (µas)')
            else:
                self.dirty_image = self.generate_dirty_image(uv_coordinates)
                self.beam_image = self.generate_beam_image(uv_coordinates)

                self.clean_image, _, _, _ = self.clean(
                    self.dirty_image,
                    self.beam_image,
                    max_iterations=10000
                )

            self.update_plots(uv_coordinates)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"Error updating results: {e}")
            print(traceback.format_exc())

    def update_plots(self, uv_coordinates):

        axes_to_clear = [self.ax_model, self.ax_beam, self.ax_uv,
                         self.ax_model_fft, self.ax_dirty, self.ax_clean]
        if self.mode_var.get() == "VLA":
            axes_to_clear.append(self.ax_array)

        for ax in axes_to_clear:
            ax.clear()

        if self.mode_var.get() == "VLA":
            self.ax_array.scatter(self.telescope_array.positions[:, 0],
                                  self.telescope_array.positions[:, 1],
                                  c='blue', marker='o')
            self.ax_array.set_title("VLA Antenna Locations")
            self.ax_array.set_xlabel("East (km)")
            self.ax_array.set_ylabel("North (km)")
            self.ax_array.grid(True)
            self.ax_array.axis('equal')
        else:
            self.plot_earth_map()

        if self.mode_var.get() == "VLA":
            max_baseline = np.max(np.linalg.norm(
                self.telescope_array.positions, axis=1))
            fov_parsecs = (206265 * (self.wavelength_var.get() * 1000) /
                           max_baseline) / 3.086e16
        else:
            fov_parsecs = (206265000 * (self.wavelength_var.get() * 1000) /
                           np.max(np.abs(uv_coordinates))) / 3.086e16

        self.ax_model.imshow(self.sky_image, cmap='hot', origin='lower',
                             extent=[-fov_parsecs / 2, fov_parsecs / 2, -fov_parsecs / 2, fov_parsecs / 2])
        self.ax_model.set_title("Model Image (Sky)")
        self.ax_model.set_xlabel("Position (parsecs)")
        self.ax_model.set_ylabel("Position (parsecs)")

        self.ax_uv.scatter(uv_coordinates[:, 0], uv_coordinates[:, 1],
                           c='blue', s=1, alpha=0.5)
        self.ax_uv.scatter(-uv_coordinates[:, 0], -uv_coordinates[:, 1],
                           c='red', s=1, alpha=0.5)
        self.ax_uv.set_title("UV Coverage")
        self.ax_uv.set_xlabel("u (wavelengths)")
        self.ax_uv.set_ylabel("v (wavelengths)")
        self.ax_uv.grid(True)
        self.ax_uv.axis('equal')

        self.ax_beam.imshow(self.beam_image, cmap='hot', origin='lower',
                            extent=[-fov_parsecs / 2, fov_parsecs / 2, -fov_parsecs / 2, fov_parsecs / 2])
        self.ax_beam.set_title("Beam Pattern (PSF)")
        self.ax_beam.set_xlabel("Position (parsecs)")
        self.ax_beam.set_ylabel("Position (parsecs)")

        self.ax_dirty.imshow(self.dirty_image, cmap='hot', origin='lower',
                             extent=[-fov_parsecs / 2, fov_parsecs / 2, -fov_parsecs / 2, fov_parsecs / 2])
        self.ax_dirty.set_title("Dirty Image")
        self.ax_dirty.set_xlabel("Position (parsecs)")
        self.ax_dirty.set_ylabel("Position (parsecs)")

        self.ax_clean.imshow(self.clean_image, cmap='hot', origin='lower',
                             extent=[-fov_parsecs / 2, fov_parsecs / 2, -fov_parsecs / 2, fov_parsecs / 2])
        self.ax_clean.set_title("CLEAN Image")
        self.ax_clean.set_xlabel("Position (parsecs)")
        self.ax_clean.set_ylabel("Position (parsecs)")

        model_fft = np.abs(fftshift(fft2(ifftshift(self.sky_image))))
        self.ax_model_fft.imshow(
            np.log10(model_fft + 1), cmap='hot', origin='lower',
            extent=[-fov_parsecs / 2, fov_parsecs / 2, -fov_parsecs / 2, fov_parsecs / 2])
        self.ax_model_fft.set_title("Model FFT")
        self.ax_model_fft.set_xlabel("Position (parsecs)")
        self.ax_model_fft.set_ylabel("Position (parsecs)")

        self.fig.subplots_adjust(hspace=0.4, wspace=0.4)
        self.fig.tight_layout()
        self.canvas.draw()

    def load_new_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.fits")])
        self.file_path = file_path
        if file_path:
            self.load_sky_image(file_path)
            self.update_results()

    def update_parameters(self):
        self.hour_angle = self.hour_angle_var.get()
        self.declination = self.declination_var.get()
        self.wavelength = self.wavelength_var.get()
        self.clean_gamma = self.clean_gamma_var.get()
        self.clean_threshold = self.clean_threshold_var.get()

        if self.mode_var.get() == "EHT":
            try:
                self.start_time = datetime.datetime.strptime(
                    self.time_var.get(), "%Y-%m-%d %H:%M:%S")
                self.duration = self.duration_var.get()
            except ValueError:
                pass

    def on_map_click(self, event):
        if self.mode_var.get() != "EHT":
            return

        if event.xdata is None or event.ydata is None:
            print("Click occurred outside the axes or on an invalid area.")
            return

        lon, lat = event.xdata, event.ydata

        lat = np.clip(lat, -90, 90)
        lon = ((lon + 180) % 360) - 180

        self.hour_angle_var.set(-lon)
        self.source_lon = lon
        self.declination_var.set(lat)

        self.update_results()

    def on_mouse_press(self, event):
        print(event)
        if event.inaxes == self.ax_array and self.mode_var.get() == "VLA":
            if self.telescope_array.select_telescope(event):
                self.update_plots(self.calculate_current_uv())
        elif event.inaxes == self.ax_map and self.mode_var.get() == "EHT":
            if self.edit_source_position_var.get() == True:
                self.on_map_click(event)
            else:
                x, y = event.xdata, event.ydata
                self.new_x = x
                self.new_y = y
                nearest_telescope, min_distance = None, 15
                telescopes = []
                for item in self.eht.tarr:
                    telescopes.append((item[0], *TelescopeConfig.ecef_to_lat_lon(item[1], item[2], item[3])))
                print(x, y)
                print(telescopes)
                for name, ty, tx in telescopes:
                    print(name, ty, tx)
                    distance = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
                    if distance < min_distance:
                        nearest_telescope = name
                        min_distance = distance
                if nearest_telescope:
                    self.selected_telescope = nearest_telescope
                print(self.selected_telescope)

    def on_mouse_motion(self, event):
        if event.inaxes and event.button == 1:
            if self.mode_var.get() == "VLA":
                if self.telescope_array.move_telescope(event):
                    print("Motion", event)
                    self.update_plots(self.calculate_current_uv())
            elif event.inaxes == self.ax_map and self.mode_var.get() == "EHT" and self.edit_source_position_var.get() == False:
                self.new_x, self.new_y = event.xdata, event.ydata

    def on_mouse_release(self, event):
        if event.inaxes == self.ax_map and self.mode_var.get() == "EHT" and self.edit_source_position_var.get() == False:
            if self.selected_telescope is not None:
                new_lat_lon = []
                for i in range(len(self.telescope_array.names)):
                    if self.telescope_array.names[i] == self.selected_telescope:
                        new_lat_lon.append((self.new_y, self.new_x))
                    else:
                        new_lat_lon.append(self.telescope_array.lat_lon[i])
                self.telescope_array.lat_lon = np.array(new_lat_lon)
                print(self.telescope_array.lat_lon)
                new_positions = [(self.selected_telescope, self.new_y, self.new_x, 0.0)]
                print(new_positions)
                self.eht = modify_telescope_positions(self.eht, new_positions)
                print(self.eht.tarr)
                self.telescope_array.selected_telescope = None
                self.selected_telescope = None
                self.update_results()
                self.plot_earth_map()

    def calculate_current_uv(self):
        mode = self.mode_var.get()
        if mode == "EHT":
            try:
                start_time = datetime.datetime.strptime(
                    self.time_var.get(), "%Y-%m-%d %H:%M:%S")
                return self.telescope_array.calculate_uv_coordinates(
                    None,
                    self.declination_var.get(),
                    start_time=start_time,
                    duration=self.duration_var.get()
                )
            except ValueError:
                print("Invalid time format")
                return None
        else:
            return self.telescope_array.calculate_uv_coordinates(
                self.hour_angle_var.get(),
                self.declination_var.get()
            )
