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
    r0 = max(0, peak_r - win); r1 = min(H - 1, peak_r + win)
    c0 = max(0, peak_c - win); c1 = min(W - 1, peak_c + win)

    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    w = np.abs(Zz[rr, cc]).astype(np.float64)
    s = float(np.sum(w))
    if s <= 1e-12:
        return float(Y[peak_r, peak_c]), float(X[peak_r, peak_c])
    lat = float(np.sum(w * Y[rr, cc]) / s)
    lon = float(np.sum(w * X[rr, cc]) / s)
    return lat, lon

def estimate_relative_depth(area_px: int, peak_abs_z: float):
    peak = max(peak_abs_z, 1e-6)
    return float(math.sqrt(max(area_px, 1)) / peak)

def run_analysis_from_tiff_bytes(
    tiff_bytes: bytes, bbox: list[float], clip_lo: float, clip_hi: float,
    smooth_on: bool, smooth_k: int, z_mode: str, thr: float, posneg: bool,
):
    Z = tiff.imread(io.BytesIO(tiff_bytes)).astype(np.float32)
    H, W = Z.shape[:2]

    X, Y = np.meshgrid(np.linspace(bbox[0], bbox[2], W), np.linspace(bbox[1], bbox[3], H))

    eps = 1e-10
    Z_db = 10.0 * np.log10(np.maximum(Z, eps))

    valid = Z_db[~np.isnan(Z_db)]
    p_lo, p_hi = np.percentile(valid, [clip_lo, clip_hi])
    Z_db_clip = np.clip(Z_db, p_lo, p_hi)

    if smooth_on and smooth_k > 1:
        # V3 - Speckle Yansıma Yokedici (Lee)
        Z_db_clip = lee_filter(Z_db_clip.astype(np.float32), size=int(smooth_k))

    if z_mode.startswith("Robust"):
        Z_z = robust_z(Z_db_clip)
    elif z_mode.startswith("Lokal"):
        Z_z = local_z_score(Z_db_clip, window_size=15)
    else:
        Z_z = classic_z(Z_db_clip)

    # O HATAYA ÇÖZÜM: Kenar yansımalarını (Sınır Çizgilerini) hedeften saymayı engeller.
    kenar = 8 
    Z_z[:kenar, :] = 0
    Z_z[-kenar:, :] = 0
    Z_z[:, :kenar] = 0
    Z_z[:, -kenar:] = 0

    if posneg:
        pos_mask = (Z_z >= thr)
        neg_mask = (Z_z <= -thr)
    else:
        pos_mask = (np.abs(Z_z) >= thr)
        neg_mask = np.zeros_like(pos_mask, dtype=bool)

    comps_pos = connected_components(pos_mask) if np.any(pos_mask) else []
    comps_neg = connected_components(neg_mask) if np.any(neg_mask) else []

    def score_components(comps, sign_label):
        ranked = []
        for comp in comps:
            pix = comp["pixels"]
            rr = np.array([p[0] for p in pix], dtype=int)
            cc = np.array([p[1] for p in pix], dtype=int)
            vals = Z_z[rr, cc]
            
            k = int(np.argmax(vals)) if sign_label == "POS" else int(np.argmin(vals))
            signed_peak = float(vals[k])
            peak_abs = float(abs(signed_peak))
            area = int(comp["area"])
            rmin, rmax, cmin, cmax = comp["bbox"]

            bbox_area = int((rmax - rmin + 1) * (cmax - cmin + 1))
            fill = (area / bbox_area) if bbox_area > 0 else 0.0
            score = peak_abs * math.log1p(area) * (0.6 + 0.8 * fill)

            target_lat, target_lon = weighted_peak_center(int(rr[k]), int(cc[k]), Z_z, X, Y, win=1)
            rel_z = estimate_relative_depth(area, peak_abs)
            
            # --- V3 AKILLI YORUM VE DERİNLİK MOTORU ---
            if rel_z < 2.5: depth_class = "Yüzeye Yakın (0-2m)"
            elif rel_z < 5.0: depth_class = "Orta Derinlik (2-5m)"
            else: depth_class = "Derin Hedef (5m+)"

            if sign_label == "POS":
                if peak_abs > 4.5 and area > 20: smart_comment = "Güçlü anomali ancak çok geniş yayılım var. Metalik yapı veya yüzey yansıması."
                elif peak_abs > 3.0 and area <= 15: smart_comment = "Kompakt ve çok güçlü hedef! Kesin donatı/metal veya yapı temeli şüphesi."
                elif fill < 0.3: smart_comment = "Parçalı yapı. Algoritma (Speckle) dalgalanması olabilir, sahada teyit edilmelidir."
                else: smart_comment = "Standart yüksek kazançlı bölge. Yüzeyde potansiyel kütle artışı."
            else:
                if peak_abs > 4.0 and area > 15: smart_comment = "Keskin negatif hacim! Derin boşluk, geniş kazı veya dev tünel karakteri taşıyor."
                elif peak_abs > 2.5 and rel_z > 4: smart_comment = "Küçük ama yapısı itibarıyla sert bir düşüş. İzole gömülü yapı/boşluk potansiyeli."
                else: smart_comment = "Standart zemin altı negatif anomali / düşük çevresel yoğunluklu bölge."

            ranked.append({
                "type": sign_label, "score": float(score), "peak_z": float(signed_peak),
                "area": area, "fill": float(fill), "bbox_rc": (int(rmin), int(rmax), int(cmin), int(cmax)),
                "target_lat": float(target_lat), "target_lon": float(target_lon),
                "rel_depth": float(rel_z), "depth_class": depth_class, "smart_comment": smart_comment,
            })
        ranked.sort(key=lambda d: d["score"], reverse=True)
        return ranked

    ranked = score_components(comps_pos, "POS") + score_components(comps_neg, "NEG")
    ranked.sort(key=lambda d: d["score"], reverse=True)

    # --- V3 BICUBIC YAZILIMSAL PÜRÜZSÜZLEŞTİRME ---
    zoom_factor = 3
    Z_db_clip_smooth = zoom(Z_db_clip, zoom_factor, order=3)
    Z_z_smooth = zoom(Z_z, zoom_factor, order=3)
    
    H_s, W_s = Z_db_clip_smooth.shape
    X_smooth, Y_smooth = np.meshgrid(np.linspace(bbox[0], bbox[2], W_s), np.linspace(bbox[1], bbox[3], H_s))
    
    pos_mask_smooth = zoom(pos_mask.astype(float), zoom_factor, order=1) >= 0.5
    neg_mask_smooth = zoom(neg_mask.astype(float), zoom_factor, order=1) >= 0.5

    return {
        "Z_db_clip": Z_db_clip_smooth, "X": X_smooth, "Y": Y_smooth, "Z_z": Z_z_smooth,
        "ranked": ranked, "pos_mask": pos_mask_smooth, "neg_mask": neg_mask_smooth,
    }
