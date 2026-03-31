import numpy as np
import matplotlib.pyplot as plt

def zscore_to_heatmap(z, threshold=2.0):
    mask = np.abs(z) >= threshold
    z_masked = np.where(mask, z, np.nan)

    fig, ax = plt.subplots()
    cax = ax.imshow(z_masked, cmap='seismic', vmin=-np.nanmax(np.abs(z)), vmax=np.nanmax(np.abs(z)))
    fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.7, label="Z-Score")
    ax.set_title("Pozitif / Negatif Anomali Heatmap")
    ax.axis('off')
    return fig
