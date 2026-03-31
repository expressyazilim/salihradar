import streamlit as st

def apply_qp_location():
    try:
        if "glat" in st.query_params and "glon" in st.query_params:
            lat = float(str(st.query_params["glat"]))
            lon = float(str(st.query_params["glon"]))
            return lat, lon
    except Exception:
        pass
    return None, None

def geolocation_button():
    # kullanıcı tıklamasıyla JS çalışsın (mobil güvenliği için şart)
    clicked = st.button("📍 Canlı Konumu Çek (mobil)", use_container_width=True)
    if not clicked:
        return

    st.components.v1.html(
        """
        <script>
        (async function(){
          try{
            if(!navigator.geolocation){ alert("Konum desteği yok."); return; }

            navigator.geolocation.getCurrentPosition(
              (pos)=>{
                const lat = pos.coords.latitude.toFixed(7);
                const lon = pos.coords.longitude.toFixed(7);

                const url = new URL(window.location.href);
                url.searchParams.set("glat", lat);
                url.searchParams.set("glon", lon);
                url.searchParams.set("gt", String(Date.now()));
                window.location.href = url.toString();
              },
              (err)=>{
                alert("Konum alınamadı: " + err.message + "\\nTelefon konum izni açık mı?");
              },
              { enableHighAccuracy:true, timeout:15000, maximumAge:0 }
            );

          }catch(e){
            alert("Konum hatası: " + e);
          }
        })();
        </script>
        """,
        height=0,
    )

    # Fallback: kullanıcıya “query param ile manuel aç” imkanı
    st.caption("Konum izni engellenirse: adres çubuğuna `?glat=41.0&glon=28.7` ekleyerek de test edebilirsin.")
