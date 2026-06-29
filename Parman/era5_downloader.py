# era5_downloader.py
import os
import xarray as xr
from pathlib import Path

def load_pat(filepath="keys.txt"):
    """Reads the PAT from a local file to avoid hardcoding secrets in scripts."""
    try:
        with open(filepath, 'r') as f:
            # Read first line and strip any whitespace/newlines
            return f.readline().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Key file '{filepath}' not found. Please create it and add your PAT.")

def fetch_era5_destine(variable, timestep='daily', bbox=None, time_slice=None, pat_file="keys.txt"):
    """
    Connects to the DestinE Earth Data Hub and lazily fetches ERA5 data.
    
    Args:
        variable (str): ERA5 variable name (e.g., 't2m', 'tp').
        timestep (str): 'hourly' or 'daily'.
        bbox (dict): Lat/Lon slicing dict, e.g., {'latitude': slice(40, 20), 'longitude': slice(250, 270)}
                     Note: ERA5 longitudes are usually 0-360. Latitudes decrease (North to South).
        time_slice (slice): e.g., slice('2020-12-01', '2021-02-28')
        pat_file (str): Path to the text file containing your token.
    
    Returns:
        xr.DataArray: The requested data slice (lazy loaded).
    """
    pat = load_pat(pat_file)
    
    if timestep == 'hourly':
        base_url = f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr"
    elif timestep == 'daily':
        base_url = f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
    else:
        raise ValueError("timestep must be 'hourly' or 'daily'")
    
    print(f"Connecting to DestinE Zarr array for {variable} ({timestep})...")
    ds = xr.open_dataset(base_url, chunks={}, engine="zarr").astype("float32")
    
    da = ds[variable]
    
    # Apply spatial slicing if provided
    if bbox:
        da = da.sel(**bbox)
        
    # Apply temporal slicing if provided
    if time_slice:
        da = da.sel(valid_time=time_slice)
        
    return da