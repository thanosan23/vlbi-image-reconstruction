import numpy as np

from config import TelescopeConfig
from itertools import combinations
import astropy.time as at
from astropy import units as u


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

                if (reading_coords and not line.startswith("#")
                        and line.strip()):
                    try:
                        pos = line.strip().split(',')
                        if len(pos) == 2:
                            e_offset = float(pos[0].strip())
                            n_offset = float(pos[1].strip())
                            antenna_positions.append(
                                (e_offset/1000, n_offset/1000))
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
        self.positions = np.array([tel["ecef"]
                                  for tel in TelescopeConfig.EHT_TELESCOPES])
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

    def calculate_uv_coordinates(self, hour_angle, declination,
                                 start_time=None, duration=None):
        if self.mode == "VLA":
            return self._calculate_vla_uv(hour_angle, declination)
        else:
            return self._calculate_eht_uv(start_time, duration, declination)

    def _calculate_vla_uv(self, hour_angle, declination):
        h0 = np.radians(hour_angle)
        delta0 = np.radians(declination)

        rotation_matrix = np.array([
            [np.sin(h0), np.cos(h0), 0],
            [-np.sin(delta0) * np.cos(h0), np.sin(delta0)
             * np.sin(h0), np.cos(delta0)],
            [np.cos(delta0) * np.cos(h0), -np.cos(delta0)
             * np.sin(h0), np.sin(delta0)]
        ])

        baselines_3d = np.hstack(
            (self.baselines, np.zeros((self.baselines.shape[0], 1))))
        uvw = np.dot(baselines_3d, rotation_matrix.T) / self.wavelength
        return uvw[:, :2]

    def _calculate_eht_uv(self, start_time, duration, declination):
        if start_time is None or duration is None:
            return self._calculate_vla_uv(0, declination)

        t_start = at.Time(start_time)

        n_samples = 24
        times = t_start + \
            np.linspace(0, duration, int(n_samples * duration)) * u.hour

        uv_points = []
        for t in times:
            lst = t.sidereal_time('apparent', longitude=0)
            ha = lst.hour - (declination / 15.0)

            h0 = np.radians(ha * 15)
            delta0 = np.radians(declination)

            rotation_matrix = np.array([
                [np.sin(h0), np.cos(h0), 0],
                [-np.sin(delta0) * np.cos(h0), np.sin(delta0)
                 * np.sin(h0), np.cos(delta0)],
                [np.cos(delta0) * np.cos(h0), -np.cos(delta0)
                 * np.sin(h0), np.sin(delta0)]
            ])

            baselines_3d = self.baselines[:, :3]
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
            self.positions[self.selected_telescope] = [
                event.xdata, event.ydata]
            self.compute_baselines()
            return True
        return False
