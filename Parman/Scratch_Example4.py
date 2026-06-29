

def run_production_analysis(
    timestep: str,
    pat: str,
    years: range,
    conditions: list,
    output_root: str,
    bbox: dict,
    logic_func, # <--- We pass the logic here!
    variable: str = 'stl1',
    variable_name: str = 'air_temperature',
    logic_params = (None, None),
    run_notes: str = ''
):
    # Setup
    if timestep == 'hourly':
        base_url = f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr"
    elif timestep == 'daily':
        base_url = f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
    else:
        raise ValueError("timestep must be 'hourly' or 'daily'")
    if logic_params is (None, None):
        raise ValueError("You must provide logic_params for the selected logic_func.")
    ds = xr.open_dataset(base_url, chunks={}, engine="zarr").astype("float32")
    out_path = Path(output_root)
    out_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        print(f"\n🌍 Processing {year} ({timestep})...")
        periods = { "early": slice(f"{year}-01-01", f"{year}-03-15"), "late":  slice(f"{year}-10-15", f"{year}-12-31") }
        
        try:
            print(f"   ⬇️  Downloading data...", end=" ")
            early_slice = ds[variable].sel(valid_time=periods['early'], **bbox).compute() - 273.15
            late_slice = ds[variable].sel(valid_time=periods['late'], **bbox).compute() - 273.15
            print("Done.")

            has_early = early_slice.valid_time.size > 0
            has_late = late_slice.valid_time.size > 0

            if not has_early and not has_late: continue     
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

        for thresh, steps in conditions:
            # === APPLY SELECTED LOGIC ===
            suitable_early = logic_func(early_slice if has_early else None, *logic_params)
            suitable_late  = logic_func(late_slice if has_late else None, *logic_params)

            coords_source = early_slice if has_early else late_slice
            
            if suitable_early is not None and suitable_late is not None:
                year_mask = np.minimum(suitable_early, suitable_late).astype('int8')
            elif suitable_early is not None:
                year_mask = suitable_early.astype('int8')
            elif suitable_late is not None:
                year_mask = suitable_late.astype('int8')
            else:
                continue

            # Save
            logic_name = logic_func.__name__.replace('calc_', '').replace('_logic', '') # e.g. "mean" or "consecutive"
            sub_folder = out_path / f"model_{logic_name}_{timestep}"
            sub_folder.mkdir(exist_ok=True)

            fname = f"{variable_name}_{year}_{thresh}C_{steps}_{logic_name}_{run_notes}.nc"

            da_out = xr.DataArray(
                [year_mask],
                coords={'time': [pd.Timestamp(f"{year}-01-01")], 'latitude': coords_source.latitude, 'longitude': coords_source.longitude}, 
                dims=('time', 'latitude', 'longitude'),
                name='suitability_flag',
                attrs={
                    'description': f"1 if SUITABLE. Model: {logic_name}",
                    'threshold_c': thresh,
                    'duration_steps': steps,
                    'year': year
                }
            )
            da_out.to_netcdf(sub_folder / fname)
            print(f"      ✅ Saved: {fname}")

    print("\n🎉 Run Complete.")













if __name__ == "__main__":
    PAT = "edh_pat_9e594c8598afc51e72214534eb56928b3fe8a54c8408f008eba5fb8e0de19bc630f0a148f5a475e15b245c85bfd3dd65"

    MY_PAT = PAT  
    CONDITIONS = [(4.4, 240)]
    CONUS_EXTENT = {'latitude': slice(50, 24), 'longitude': slice(235, 294)}
    SLOPE = 0.127
    INTERCEPT = 0.316
    # RUN 1: The original "Consecutive" model
    print("🚀 Starting Run 1: Consecutive Model")
    run_production_analysis(
        timestep='hourly', pat=MY_PAT, years=range(2004, 2025),
        conditions=CONDITIONS, output_root="./outputs/v2_outputs/air_10recover", bbox=CONUS_EXTENT,
        variable='t2m', variable_name='air_temp',
        logic_func=calc_lt99_logic,
        logic_params=(SLOPE, INTERCEPT),
        run_notes='10_recovery'
    )