import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from itertools import combinations
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter
from mpl_toolkits.basemap import Basemap
import datetime
import astropy.time as at
import astropy.coordinates as coord
from astropy import units as u

class TelescopeConfig:
    VLA_TELESCOPES = {
        "A": "arrays/VLA_A.config",
        "B": "arrays/VLA_B.config",
        "C": "arrays/VLA_C.config",
        "D": "arrays/VLA_D.config"
    }
    
    VLA_LATITUDE = 34.078745
    
    EHT_TELESCOPES = [
        {"name": "PV", "ecef": (5088967.9, -301681.6, 3825015.8)},
        {"name": "SMT", "ecef": (-1828796.2, -5054406.8, 3427865.2)},
        {"name": "SMA", "ecef": (-5464523.4, -2493147.08, 2150611.75)},
        {"name": "LMT", "ecef": (-768713.9637, -5988541.7982, 2063275.9472)},
        {"name": "ALMA", "ecef": (2225061.164, -5440057.37, -2481681.15)},
        {"name": "SPT", "ecef": (0.01, 0.01, -6359609.7)},
        {"name": "APEX", "ecef": (2225039.53, -5441197.63, -2479303.36)},
        {"name": "JCMT", "ecef": (-5464584.68, -2493001.17, 2150653.98)},
    ]

    @staticmethod
    def ecef_to_lat_lon(x, y, z):
        a = 6378137.0  # semi-major axis in meters
        e2 = 0.00669437999014  # eccentricity squared
        lon = np.degrees(np.arctan2(y, x))
        p = np.sqrt(x**2 + y**2)
        lat = np.degrees(np.arctan2(z, p * (1 - e2)))
        return lat, lon

class TelescopeArray:
    def __init__(self, mode="VLA"):
        self.mode = mode
        self.positions = None
        self.baselines = None
        self.wavelength = 1.0
        self.names = None
        self.lat_lon = None
        self.latitude = TelescopeConfig.VLA_LATITUDE
        self.selected_telescope = None
        self.original_positions = None
        
        if mode == "VLA":
            self.load_vla_config()
        else:
            self.load_eht_config()
    
    def parse_config_file(self, file_path):
        antenna_positions = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            reading_coords = False
            for line in lines:
                if "offset E, offset N" in line:
                    reading_coords = True
                    continue
                
                if reading_coords and not line.startswith("#") and line.strip():
                    try:
                        pos = line.strip().split(',')
                        if len(pos) == 2:
                            e_offset = float(pos[0].strip())
                            n_offset = float(pos[1].strip())
                            antenna_positions.append((e_offset/1000, n_offset/1000)) 
                    except ValueError:
                        continue
        return np.array(antenna_positions)

    def load_vla_config(self):
        config_file = TelescopeConfig.VLA_TELESCOPES["A"]
        self.positions = self.parse_config_file(config_file)
        if len(self.positions) > 0:
            self.compute_baselines()
        else:
            print("Warning: No antenna positions loaded from config file")

    def load_eht_config(self):
        self.positions = np.array([tel["ecef"] for tel in TelescopeConfig.EHT_TELESCOPES])
        self.names = [tel["name"] for tel in TelescopeConfig.EHT_TELESCOPES]
        self.lat_lon = np.array([TelescopeConfig.ecef_to_lat_lon(*pos) 
                                for pos in self.positions])
        self.compute_baselines()
        
    def compute_baselines(self):
        if self.positions is None or len(self.positions) == 0:
            print("Warning: No positions available to compute baselines")
            return
            
        self.baselines = []
        for (i, pos1), (j, pos2) in combinations(enumerate(self.positions), 2):
            self.baselines.append(pos2 - pos1)
        self.baselines = np.array(self.baselines)

    def calculate_uv_coordinates(self, hour_angle, declination, start_time=None, duration=None):
        if self.mode == "VLA":
            return self._calculate_vla_uv(hour_angle, declination)
        else:
            return self._calculate_eht_uv(start_time, duration, declination)

    def _calculate_vla_uv(self, hour_angle, declination):
        h0 = np.radians(hour_angle)
        delta0 = np.radians(declination)
        
        rotation_matrix = np.array([
            [np.sin(h0), np.cos(h0), 0],
            [-np.sin(delta0) * np.cos(h0), np.sin(delta0) * np.sin(h0), np.cos(delta0)],
            [np.cos(delta0) * np.cos(h0), -np.cos(delta0) * np.sin(h0), np.sin(delta0)]
        ])
        
        baselines_3d = np.hstack((self.baselines, np.zeros((self.baselines.shape[0], 1))))
        uvw = np.dot(baselines_3d, rotation_matrix.T) / self.wavelength
        return uvw[:, :2]

    def _calculate_eht_uv(self, start_time, duration, declination):
        if start_time is None or duration is None:
            return self._calculate_vla_uv(0, declination)
        
        t_start = at.Time(start_time)
        
        n_samples = 24  # Samples per hour
        times = t_start + np.linspace(0, duration, int(n_samples * duration)) * u.hour
        
        uv_points = []
        for t in times:
            lst = t.sidereal_time('apparent', longitude=0)
            ha = lst.hour - (declination / 15.0)  # Convert dec to hours
            
            h0 = np.radians(ha * 15)  # Convert hour angle to radians
            delta0 = np.radians(declination)
            
            rotation_matrix = np.array([
                [np.sin(h0), np.cos(h0), 0],
                [-np.sin(delta0) * np.cos(h0), np.sin(delta0) * np.sin(h0), np.cos(delta0)],
                [np.cos(delta0) * np.cos(h0), -np.cos(delta0) * np.sin(h0), np.sin(delta0)]
            ])
            
            baselines_3d = self.baselines[:, :3]  # Take only x, y, z components
            uvw = np.dot(baselines_3d, rotation_matrix.T) / self.wavelength
            uv_points.append(uvw[:, :2])
        
        return np.vstack(uv_points)

    def select_telescope(self, event):
        if not hasattr(self, 'positions'):
            return
            
        click_pos = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.positions - click_pos, axis=1)
        nearest = np.argmin(distances)
        
        if distances[nearest] < 0.5:  
            self.selected_telescope = nearest
            return True
        return False
    
    def move_telescope(self, event):
        if self.selected_telescope is not None:
            self.positions[self.selected_telescope] = [event.xdata, event.ydata]
            self.compute_baselines()
            return True
        return False

class TelescopeApp:
    def __init__(self, root):
        self.root = root
        self.current_mode = "VLA"
        
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        self.image_size = 256
        self.wavelength = 1.0
        self.hour_angle = 45
        self.declination = 34.078745
        self.clean_gamma = 0.1
        self.clean_threshold = 0.001
        self.start_time = datetime.datetime.now()
        
        self.mode_var = tk.StringVar(value="VLA")
        self.wavelength_var = tk.DoubleVar(value=self.wavelength)
        self.hour_angle_var = tk.DoubleVar(value=self.hour_angle)
        self.declination_var = tk.DoubleVar(value=self.declination)
        self.clean_gamma_var = tk.DoubleVar(value=self.clean_gamma)
        self.clean_threshold_var = tk.DoubleVar(value=self.clean_threshold)
        self.time_var = tk.StringVar(value=self.start_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.duration_var = tk.DoubleVar(value=12.0)
        
        self.telescope_array = TelescopeArray(mode="VLA")
        
        control_panel = ttk.Frame(main_container)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.setup_telescope_management(control_panel)
        
        self.setup_control_sections(control_panel)
        
        plot_panel = ttk.Frame(main_container)
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_plots(plot_panel)
        
        self.load_sky_image("models/double_wide.png")
        self.update_results()

    def setup_telescope_management(self, parent):
        mgmt_frame = ttk.LabelFrame(parent, text="Telescope Management", padding="5")
        mgmt_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.config_frame = ttk.Frame(mgmt_frame)
        self.config_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.config_frame, text="Array Config:").pack(side=tk.LEFT)
        self.config_var = tk.StringVar(value="A")
        config_cb = ttk.Combobox(self.config_frame, textvariable=self.config_var,
                                values=list(TelescopeConfig.VLA_TELESCOPES.keys()))
        config_cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.telescope_list = tk.Listbox(mgmt_frame, height=15, width=30, selectmode=tk.SINGLE)
        self.telescope_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_frame = ttk.Frame(mgmt_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add Telescope", 
                   command=self.add_telescope).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Selected",
                   command=self.remove_telescope).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(mgmt_frame, text="Load Sky Image",
                   command=self.load_new_image).pack(fill=tk.X, pady=5)
        
        self.update_telescope_list()

    def update_telescope_list(self):
        self.telescope_list.delete(0, tk.END)
        if self.mode_var.get() == "VLA":
            for i, pos in enumerate(self.telescope_array.positions):
                self.telescope_list.insert(tk.END, 
                    f"Telescope {i}: E={pos[0]:.2f}, N={pos[1]:.2f}")
        else:
            for i, (name, (lat, lon)) in enumerate(zip(
                self.telescope_array.names, self.telescope_array.lat_lon)):
                self.telescope_list.insert(tk.END, 
                    f"{name}: Lat={lat:.2f}, Lon={lon:.2f}")

    def add_telescope(self):
        if self.mode_var.get() == "VLA":
            file_path = filedialog.askopenfilename(
                initialdir="arrays",
                title="Select Array Configuration",
                filetypes=[("Config files", "*.config")]
            )
            if file_path:
                new_positions = self.telescope_array.parse_config_file(file_path)
                if new_positions is not None:
                    self.telescope_array.positions = new_positions
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
                self.telescope_array.lat_lon = np.vstack([
                    self.telescope_array.lat_lon, new_lat_lon
                ])
            self.telescope_array.compute_baselines()
            self.update_telescope_list()
            self.update_results()

    def remove_telescope(self):
        selection = self.telescope_list.curselection()
        if not selection:
            return
        
        idx = selection[0]
        if self.mode_var.get() == "VLA":
            if len(self.telescope_array.positions) > 1:
                self.telescope_array.positions = np.delete(
                    self.telescope_array.positions, idx, axis=0)
                self.telescope_array.compute_baselines()
                self.update_telescope_list()
                self.update_results()
        else:
            if len(self.telescope_array.names) > 1:
                self.telescope_array.names.pop(idx)
                self.telescope_array.lat_lon = np.delete(
                    self.telescope_array.lat_lon, idx, axis=0)
                self.telescope_array.compute_baselines()
                self.update_telescope_list()
                self.update_results()

    def load_new_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.fits")])
        if file_path:
            self.load_sky_image(file_path)
            self.update_results()

    def setup_control_sections(self, parent):
        style = ttk.Style()
        style.configure('Modern.TLabelframe', padding=10)
        style.configure('Modern.TButton', padding=5)
        style.configure('Horizontal.TScale', background='#f0f0f0')
        
        mode_frame = ttk.LabelFrame(parent, text="Array Mode", style='Modern.TLabelframe')
        mode_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.mode_var = tk.StringVar(value=self.current_mode)
        ttk.Radiobutton(mode_frame, text="VLA", variable=self.mode_var, 
                       value="VLA", command=self.change_mode).pack(side=tk.LEFT, padx=20, pady=5)
        ttk.Radiobutton(mode_frame, text="EHT", variable=self.mode_var, 
                       value="EHT", command=self.change_mode).pack(side=tk.LEFT, padx=20, pady=5)
        
        array_frame = ttk.LabelFrame(parent, text="Array Parameters", style='Modern.TLabelframe')
        array_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        wavelength_frame = ttk.Frame(array_frame)
        wavelength_frame.pack(fill=tk.X, pady=5)
        ttk.Label(wavelength_frame, text="Wavelength (m):").pack(side=tk.LEFT)
        self.wavelength_slider = ttk.Scale(wavelength_frame, from_=0.1, to=10.0, 
                                         variable=self.wavelength_var, orient=tk.HORIZONTAL)
        self.wavelength_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        self.vla_frame = ttk.Frame(array_frame)
        ttk.Label(self.vla_frame, text="Hour Angle:").pack(side=tk.LEFT)
        self.hour_angle_slider = ttk.Scale(self.vla_frame, from_=-180, to=180,
                                         variable=self.hour_angle_var, orient=tk.HORIZONTAL)
        self.hour_angle_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        self.eht_frame = ttk.Frame(array_frame)
        
        time_frame = ttk.Frame(self.eht_frame)
        time_frame.pack(fill=tk.X, pady=5)
        ttk.Label(time_frame, text="Start Time:").pack(side=tk.LEFT)
        self.time_label = ttk.Label(time_frame, textvariable=self.time_var)
        self.time_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(time_frame, text="Set Time", 
                   command=self.set_time).pack(side=tk.RIGHT)
        
        duration_frame = ttk.Frame(self.eht_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(duration_frame, text="Duration (hrs):").pack(side=tk.LEFT)
        self.duration_slider = ttk.Scale(duration_frame, from_=1, to=24,
                                       variable=self.duration_var, orient=tk.HORIZONTAL)
        self.duration_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        dec_frame = ttk.Frame(array_frame)
        dec_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dec_frame, text="Declination:").pack(side=tk.LEFT)
        self.dec_slider = ttk.Scale(dec_frame, from_=-90, to=90,
                                   variable=self.declination_var, orient=tk.HORIZONTAL)
        self.dec_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        clean_frame = ttk.LabelFrame(parent, text="CLEAN Parameters", style='Modern.TLabelframe')
        clean_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        gamma_frame = ttk.Frame(clean_frame)
        gamma_frame.pack(fill=tk.X, pady=5)
        ttk.Label(gamma_frame, text="Loop Gain:").pack(side=tk.LEFT)
        self.gamma_slider = ttk.Scale(gamma_frame, from_=0.01, to=1.0,
                                     variable=self.clean_gamma_var, orient=tk.HORIZONTAL)
        self.gamma_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        threshold_frame = ttk.Frame(clean_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        ttk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0.0001, to=0.1,
                                        variable=self.clean_threshold_var, orient=tk.HORIZONTAL)
        self.threshold_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        update_frame = ttk.Frame(parent)
        update_frame.pack(fill=tk.X, pady=10, padx=5)
        self.update_button = ttk.Button(update_frame, text="Update Results",
                                      command=self.update_results, style='Modern.TButton')
        self.update_button.pack(fill=tk.X, pady=5)

    def set_time(self):
        current_time = datetime.datetime.strptime(self.time_var.get(), "%Y-%m-%d %H:%M:%S")
        
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
        
        ttk.Entry(time_frame, textvariable=hour_var, width=3).pack(side=tk.LEFT)
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

        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.setup_telescope_management(self.canvas.get_tk_widget().master)

    def change_mode(self):
        mode = self.mode_var.get()
        self.telescope_array = TelescopeArray(mode=mode)
        
        for widget in self.canvas.get_tk_widget().master.winfo_children():
            widget.destroy()
        
        self.setup_plots(self.canvas.get_tk_widget().master)
        
        self.update_mode_controls()
        self.update_results()

    def update_mode_controls(self):
        mode = self.mode_var.get()
        if mode == "VLA":
            if hasattr(self, 'eht_frame'):
                self.eht_frame.pack_forget()
            self.vla_frame.pack(fill=tk.X, pady=2)
        else:
            self.vla_frame.pack_forget()
            self.eht_frame.pack(fill=tk.X, pady=2)

    def load_sky_image(self, image_path):
        img = Image.open(image_path).convert("L")
        img = img.resize((self.image_size, self.image_size))
        self.sky_image = np.array(img) / 255.0

    def generate_dirty_image(self, uv_coordinates):
        ft_image = fftshift(fft2(ifftshift(self.sky_image)))
        visibilities = []
        image_center = np.array(self.sky_image.shape) // 2
        
        max_uv = np.max(np.abs(uv_coordinates))
        scale_factor = (self.image_size // 4) / max_uv
        
        for u, v in uv_coordinates:
            u_scaled = int(round(u * scale_factor)) + image_center[0]
            v_scaled = int(round(v * scale_factor)) + image_center[1]
            if 0 <= u_scaled < self.image_size and 0 <= v_scaled < self.image_size:
                visibilities.append(ft_image[u_scaled, v_scaled])
            else:
                visibilities.append(0)
        
        visibilities = np.array(visibilities)
        ft_dirty_image = np.zeros((self.image_size, self.image_size), dtype=complex)
        
        for (u, v), vis in zip(uv_coordinates, visibilities):
            u_scaled = int(round(u * scale_factor)) + image_center[0]
            v_scaled = int(round(v * scale_factor)) + image_center[1]
            if 0 <= u_scaled < self.image_size and 0 <= v_scaled < self.image_size:
                ft_dirty_image[u_scaled, v_scaled] = vis
                ft_dirty_image[-u_scaled % self.image_size, -v_scaled % self.image_size] = np.conj(vis)

        dirty_image = np.abs(ifftshift(ifft2(fftshift(ft_dirty_image))))
        return dirty_image

    def generate_beam_image(self, uv_coordinates):
        ft_beam_image = np.zeros((self.image_size, self.image_size), dtype=complex)
        image_center = np.array((self.image_size // 2, self.image_size // 2))
        
        max_uv = np.max(np.abs(uv_coordinates))
        scale_factor = (self.image_size // 4) / max_uv
        
        for u, v in uv_coordinates:
            u_scaled = int(round(u * scale_factor)) + image_center[0]
            v_scaled = int(round(v * scale_factor)) + image_center[1]
            
            if 0 <= u_scaled < self.image_size and 0 <= v_scaled < self.image_size:
                ft_beam_image[u_scaled, v_scaled] = 1
                ft_beam_image[-u_scaled % self.image_size, -v_scaled % self.image_size] = 1

        beam_image = np.abs(ifftshift(ifft2(fftshift(ft_beam_image))))
        beam_image /= np.max(beam_image)
        return beam_image

    def clean(self, dirty_image, beam_image, max_iterations=100):
        cleaned = np.zeros_like(dirty_image)
        residual = dirty_image.copy()
        beam_max = np.max(beam_image)
        beam_center = np.array(beam_image.shape) // 2
        
        components = []
        
        for iteration in range(max_iterations):
            max_pos = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
            max_val = residual[max_pos]
            
            if np.abs(max_val) < self.clean_threshold * beam_max:
                break
            
            components.append((max_pos, max_val * self.clean_gamma))
            
            y_start = max_pos[0] - beam_center[0]
            y_end = y_start + beam_image.shape[0]
            x_start = max_pos[1] - beam_center[1]
            x_end = x_start + beam_image.shape[1]
            
            y_slice = slice(max(0, y_start), min(residual.shape[0], y_end))
            x_slice = slice(max(0, x_start), min(residual.shape[1], x_end))
            beam_y_slice = slice(max(0, -y_start), min(beam_image.shape[0], residual.shape[0] - y_start))
            beam_x_slice = slice(max(0, -x_start), min(beam_image.shape[1], residual.shape[1] - x_start))
            
            residual[y_slice, x_slice] -= max_val * self.clean_gamma * \
                beam_image[beam_y_slice, beam_x_slice]
        
        clean_beam = gaussian_filter(beam_image, sigma=2)
        clean_beam /= np.max(clean_beam)
        
        for (pos, amp) in components:
            y_start = pos[0] - beam_center[0]
            y_end = y_start + clean_beam.shape[0]
            x_start = pos[1] - beam_center[1]
            x_end = x_start + clean_beam.shape[1]
            
            y_slice = slice(max(0, y_start), min(cleaned.shape[0], y_end))
            x_slice = slice(max(0, x_start), min(cleaned.shape[1], x_end))
            beam_y_slice = slice(max(0, -y_start), min(clean_beam.shape[0], cleaned.shape[0] - y_start))
            beam_x_slice = slice(max(0, -x_start), min(clean_beam.shape[1], cleaned.shape[1] - x_start))
            
            cleaned[y_slice, x_slice] += amp * clean_beam[beam_y_slice, beam_x_slice]
        
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
        
        for name, (lat, lon) in zip(self.telescope_array.names, self.telescope_array.lat_lon):
            x, y = m(lon, lat)
            m.plot(x, y, 'ro', markersize=6)
            self.ax_map.text(x + 2, y + 2, name, fontsize=8)
        
        for i, (lat1, lon1) in enumerate(self.telescope_array.lat_lon):
            for lat2, lon2 in self.telescope_array.lat_lon[i+1:]:
                x1, y1 = m(lon1, lat1)
                x2, y2 = m(lon2, lat2)
                self.ax_map.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=0.5)
        
        dec = self.declination_var.get()
        if -90 <= dec <= 90:
            lons = np.linspace(-180, 180, 360)
            decs = np.ones_like(lons) * dec
            x, y = m(lons, decs)
            m.plot(x, y, 'g--', alpha=0.5, label='Source declination')
            
            if hasattr(self, 'source_lon'):
                x, y = m(self.source_lon, dec)
                m.plot(x, y, 'g*', markersize=10, label='Source')
        
        self.ax_map.set_title("EHT Telescope Locations\nClick to set source position")
        self.ax_map.grid(True)
        
        m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,181.,60.), labels=[0,0,0,1])

    def update_results(self):
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
                    uv_coordinates = self.telescope_array.calculate_uv_coordinates(
                        None,
                        self.declination_var.get(),
                        start_time=start_time,
                        duration=self.duration_var.get()
                    )
                except ValueError:
                    print("Invalid time format")
                    return
            
            self.telescope_array.wavelength = self.wavelength_var.get()
            
            self.dirty_image = self.generate_dirty_image(uv_coordinates)
            self.beam_image = self.generate_beam_image(uv_coordinates)
            self.clean_image, _, _, _ = self.clean(
                self.dirty_image, 
                self.beam_image,
                max_iterations=100
            )
            
            self.update_plots(uv_coordinates)
            
        except Exception as e:
            print(f"Error updating results: {e}")

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
            max_baseline = np.max(np.linalg.norm(self.telescope_array.positions, axis=1))
            fov = 206265 * self.wavelength_var.get() / max_baseline  # arcseconds
            scale_unit = "arcsec"
        else:
            fov = 206265000 * self.wavelength_var.get() / np.max(np.abs(uv_coordinates))
            scale_unit = "μarcsec"

        self.ax_model.imshow(self.sky_image, cmap='hot', origin='lower')
        scale_length = fov / 4
        bar_x = self.image_size * 0.1
        bar_y = self.image_size * 0.1
        bar_width = self.image_size * 0.1
        self.ax_model.plot([bar_x, bar_x + bar_width], [bar_y, bar_y], 'w-', linewidth=1)
        self.ax_model.text(bar_x, bar_y - self.image_size * 0.08, 
                         f'{scale_length:.1f} {scale_unit}', 
                         color='white', fontsize=8)
        self.ax_model.set_title("Model Image (Sky)")

        self.ax_beam.imshow(self.beam_image, cmap='hot', origin='lower')
        self.ax_beam.set_title("Beam Pattern (PSF)")

        self.ax_uv.scatter(uv_coordinates[:, 0], uv_coordinates[:, 1], 
                          c='blue', s=1, alpha=0.5)
        self.ax_uv.scatter(-uv_coordinates[:, 0], -uv_coordinates[:, 1], 
                          c='red', s=1, alpha=0.5)
        self.ax_uv.set_title("UV Coverage")
        self.ax_uv.set_xlabel("u (kλ)")
        self.ax_uv.set_ylabel("v (kλ)")
        self.ax_uv.grid(True)
        self.ax_uv.axis('equal')

        model_fft = np.abs(fftshift(fft2(ifftshift(self.sky_image))))
        self.ax_model_fft.imshow(np.log10(model_fft + 1), cmap='hot', origin='lower')
        self.ax_model_fft.set_title("Model FFT")

        self.ax_dirty.imshow(self.dirty_image, cmap='hot', origin='lower')
        self.ax_dirty.set_title("Dirty Image")

        self.ax_clean.imshow(self.clean_image, cmap='hot', origin='lower')
        self.ax_clean.set_title("CLEAN Image")

        self.fig.tight_layout()
        self.canvas.draw()

    def load_new_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.fits")])
        if file_path:
            self.load_sky_image(file_path)
            self.update_results()

    def update_parameters(self):
        self.hour_angle = self.hour_angle_var.get()
        self.declination = self.declination_var.get()
        self.wavelength = self.wavelength_var.get()
        self.clean_gamma = self.gamma_var.get()
        self.clean_threshold = self.threshold_var.get()
        
        if self.mode_var.get() == "EHT":
            try:
                self.start_time = datetime.datetime.strptime(
                    self.time_entry.get(), "%Y-%m-%d %H:%M:%S")
                self.duration = self.duration_var.get()
            except ValueError:
                pass

    def on_map_click(self, event):
        if self.mode_var.get() != "EHT" or event.inaxes != self.ax_map:
            return
        
        lon, lat = event.xdata, event.ydata
        
        lat = np.clip(lat, -90, 90)
        lon = ((lon + 180) % 360) - 180
        
        self.source_lon = lon
        
        self.declination_var.set(lat)
        
        self.update_results()

    def on_mouse_press(self, event):
        if event.inaxes == self.ax_array and self.mode_var.get() == "VLA":
            if self.telescope_array.select_telescope(event):
                self.update_plots(self.calculate_current_uv())
        elif event.inaxes == self.ax_map and self.mode_var.get() == "EHT":
            self.on_map_click(event)

    def on_mouse_motion(self, event):
        if event.inaxes and event.button == 1: 
            if self.telescope_array.move_telescope(event):
                self.update_plots(self.calculate_current_uv())

    def on_mouse_release(self, event):
        if self.telescope_array.selected_telescope is not None:
            self.telescope_array.selected_telescope = None
            self.update_results()

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

def main():
    root = tk.Tk()
    root.title("Radio Interferometer Simulator")
    
    style = ttk.Style()
    style.theme_use('default') 
    
    window_width = 1400
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    app = TelescopeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()