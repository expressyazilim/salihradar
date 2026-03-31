import requests
import streamlit as st

AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

@st.cache_data(ttl=45 * 60, show_spinner=False)
def cached_token(client_id: str, client_secret: str, username: str | None = None, password: str | None = None):
    # 1) client_credentials
    try:
        data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
        r = requests.post(AUTH_URL, data=data, timeout=30)
        if r.status_code == 200:
            return r.json().get("access_token")
    except Exception:
        pass

    # 2) password grant (opsiyonel)
    if username and password:
        try:
            data = {"grant_type": "password", "client_id": client_id, "username": username, "password": password}
            r = requests.post(AUTH_URL, data=data, timeout=30)
            if r.status_code == 200:
                return r.json().get("access_token")
        except Exception:
            pass

    return None

def get_token_from_secrets() -> str:
    if "CDSE_CLIENT_ID" not in st.secrets or "CDSE_CLIENT_SECRET" not in st.secrets:
        raise RuntimeError("Secrets eksik: CDSE_CLIENT_ID ve CDSE_CLIENT_SECRET gerekli.")
    token = cached_token(
        st.secrets["CDSE_CLIENT_ID"],
        st.secrets["CDSE_CLIENT_SECRET"],
        st.secrets.get("CDSE_USERNAME"),
        st.secrets.get("CDSE_PASSWORD"),
    )
    if not token:
        raise RuntimeError("Token alınamadı (client_id/secret yanlış olabilir).")
    return token

@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_s1_tiff_bytes(token: str, bbox: list[float], width: int, height: int) -> bytes:
    # Sentinel-1 GRD VV
    evalscript = """
    function setup() {
      return { input: ["VV"], output: { id: "default", bands: 1, sampleType: "FLOAT32" } };
    }
    function evaluatePixel(sample) { return [sample.VV]; }
    """
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-1-grd"}],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    res = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=80,
    )
    if res.status_code != 200:
        raise RuntimeError(f"CDSE HTTP {res.status_code} | {res.text[:400]}")
    return res.content
