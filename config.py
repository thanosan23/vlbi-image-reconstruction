
import numpy as np

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

    @staticmethod
    def ecef_to_lat_lon_altitude(x, y, z):
        a = 6378137.0  # semi-major axis in meters
        e2 = 0.00669437999014  # eccentricity squared
        lon = np.degrees(np.arctan2(y, x))
        p = np.sqrt(x**2 + y**2)
        lat = np.degrees(np.arctan2(z, p * (1 - e2)))
        lat_rad = np.radians(lat)
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        altitude = p / np.cos(lat_rad) - N
        return lat, lon, altitude