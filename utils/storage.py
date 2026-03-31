import json
import os
from datetime import datetime

HISTORY_FILE = "scan_history.json"

def _read_history_raw():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return []
            data = json.loads(txt)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def load_history() -> list[dict]:
    raw = _read_history_raw()

    out = []
    for it in raw:
        if not isinstance(it, dict):
            continue

        # eski sürümlerle uyumluluk (timestamp/ts vs.)
        ts = it.get("ts") or it.get("timestamp") or "Tarih yok"
        lat = it.get("lat") if it.get("lat") is not None else it.get("latitude")
        lon = it.get("lon") if it.get("lon") is not None else it.get("longitude")

        out.append({
            "name": it.get("name") or it.get("scan_name") or "",
            "ts": ts,
            "lat": lat,
            "lon": lon,
            "cap_m": it.get("cap_m") if it.get("cap_m") is not None else it.get("cap"),
            "thr": it.get("thr") if it.get("thr") is not None else it.get("threshold"),
            "z_mode": it.get("z_mode"),
            "top": it.get("top") or [],
        })
    return out

def append_history(*, name: str, lat: float, lon: float, cap_m: int, thr: float, z_mode: str, top: list[dict]):
    # None yazılmasını tamamen engelle
    rec = {
        "name": str(name or ""),
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lat": float(lat),
        "lon": float(lon),
        "cap_m": int(cap_m),
        "thr": float(thr),
        "z_mode": str(z_mode),
        "top": top,
    }

    hist = load_history()
    hist.append(rec)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)
