import math
import io
import numpy as np
import tifffile as tiff
from collections import deque
from scipy.ndimage import uniform_filter, zoom

def parse_coord_pair(s: str):
    if not s:
        return None, None
    s = s.strip().replace(",", " ")
    parts = [p for p in s.split() if p.strip()]
    if len(parts) < 2:
        return None, None
    try:
        lat = float(parts[0].replace(",", "."))
        lon = float(parts[1].replace(",", "."))
        return lat, lon
    except Exception:
        return None, None

def bbox_from_latlon(lat: float, lon: float, cap_m: float):
    lat_f = cap_m / 111320.0
    lon_f = cap_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    return [lon - lon_f, lat - lat_f, lon + lon_f, lat + lat_f]

def lee_filter(img: np.ndarray, size: int):
    # Uydu verisi pürüz giderici SAR Lee-Filter
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-8)
    return img_mean + img_weights * (img - img_mean)

def robust_z(x: np.ndarray):
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return x * np.nan
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    denom = (1.4826 * mad) if mad > 1e-9 else (np.std(valid) if np.std(valid) > 1e-9 else 1.0)
    return (x - med) / denom

def classic_z(x: np.ndarray):
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return x * np.nan
    mu = float(np.mean(valid))
    sd = float(np.std(valid)) if float(np.std(valid)) > 1e-9 else 1.0
    return (x - mu) / sd

def local_z_score(matrix: np.ndarray, window_size: int = 15):
    local_mean = uniform_filter(matrix, size=window_size)
    local_sq_mean = uniform_filter(matrix**2, size=window_size)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0) + 1e-8)
    return (matrix - local_mean) / local_std

def connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),       ( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                pixels = []
                rmin=rmax=r
                cmin=cmax=c
                while q:
                    rr, cc = q.popleft()
                    pixels.append((rr, cc))
                    rmin = min(rmin, rr); rmax = max(rmax, rr)
                    cmin = min(cmin, cc); cmax = max(cmax, cc)
                    for dr, dc in neighbors:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                comps.append({"pixels": pixels, "area": len(pixels), "bbox": (rmin, rmax, cmin, cmax)})
    return comps

def weighted_peak_center(peak_r, peak_c, Zz, X, Y, win=1):
    H, W = Zz.shape
    r0 = max(0, peak_r - win); r1 = min(H - 
