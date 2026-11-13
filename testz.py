import os
import re
import sys
import math
import time
import sqlite3
import argparse
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")




# ==============================================================================
# SECTION 0: COORDINATE, EXAMPLE MATCHING & CLI INPUTS
# ==============================================================================
def get_user_inputs() -> tuple[float, float, str, str, int | None, bool]:
    parser = argparse.ArgumentParser(
        description="SiriusTa: Initial (LON, LAT) vs Matched Zone Example (LON_ex, LAT_ex) + result_com.csv"
    )
    parser.add_argument("--lat", type=float, help="Initial Target Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, help="Initial Target Longitude (-180 to 180)")
    parser.add_argument("--name", type=str, default="", help="Optional initial plot label")
    parser.add_argument("--prev-crop", type=str, default="", help="Previous harvested crop (e.g., maize, cotton, none)")
    parser.add_argument("--ex-row", type=int, default=None, help="Optional row # (1-12) from 12_ex.csv to force example")
    parser.add_argument("--no-show", action="store_true", help="Save PNG without opening plot window")
    args = parser.parse_args()


    print("=" * 82)
    print("SIRIUSTA — DUAL-COORDINATE (TARGET vs MATCHED ZONE EXAMPLE) COMPARISON ENGINE")
    print("=" * 82)


    lat = args.lat
    lon = args.lon


    if lat is None or lon is None:
        try:
            if lat is None:
                lat = float(input("Enter Initial Latitude  (e.g., 30.2200): ").strip())
            if lon is None:
                lon = float(input("Enter Initial Longitude (e.g., 31.3750): ").strip())
        except ValueError:
            print("\n[INPUT ERROR] Latitude and Longitude must be valid numeric values.")
            sys.exit(1)


    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        print("\n[INPUT ERROR] Coordinates out of Earth bounds (-90..90, -180..180).")
        sys.exit(1)


    plot_name = args.name.strip()
    if not plot_name and args.lat is None:
        plot_name = input("Enter Initial Plot Label (optional, press Enter to skip): ").strip()
    if not plot_name:
        plot_name = f"Target Plot ({lat}, {lon})"


    prev_crop = args.prev_crop.strip()
    if not prev_crop and args.lat is None:
        prev_crop = input("Previous Harvested Crop (e.g., maize, cotton, none) [default: none]: ").strip()
    if not prev_crop:
        prev_crop = "none"


    return lat, lon, plot_name, prev_crop, args.ex_row, args.no_show




def load_local_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Loads:
      1. SiriusTa_AEZXX_MJO_Dataset.csv
      2. SiriusTa_AEZXX_MJO_90_Matrix.csv
      3. 12_ex.csv (Matched Zone Examples Table)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [base_dir, os.path.join(base_dir, "data"), os.path.join(base_dir, "dataset")]


    def find_file(filename: str) -> str | None:
        for folder in search_dirs:
            cand = os.path.join(folder, filename)
            if os.path.isfile(cand):
                return cand
        return None


    ds_path = find_file("SiriusTa_AEZXX_MJO_Dataset.csv")
    mat_path = find_file("SiriusTa_AEZXX_MJO_90_Matrix.csv")
    ex_path = find_file("12_ex.csv")


    if not ds_path or not mat_path or not ex_path:
        print(
            f"\n[DATASET ERROR] Required files missing in {base_dir}:\n"
            f"  1. SiriusTa_AEZXX_MJO_Dataset.csv -> {'FOUND' if ds_path else 'MISSING'}\n"
            f"  2. SiriusTa_AEZXX_MJO_90_Matrix.csv -> {'FOUND' if mat_path else 'MISSING'}\n"
            f"  3. 12_ex.csv                        -> {'FOUND' if ex_path else 'MISSING'}\n"
            f"Place all three CSV files in the same folder as testb.py and re-run."
        )
        sys.exit(1)


    df_ds = pd.read_csv(ds_path)
    df_mat = pd.read_csv(mat_path)
    df_ex = pd.read_csv(ex_path)


    df_ds.columns = [str(c).strip() for c in df_ds.columns]
    df_mat.columns = [str(c).strip() for c in df_mat.columns]
    df_ex.columns = [str(c).strip() for c in df_ex.columns]
    df_ds["label"] = df_ds["label"].astype(str).str.strip().str.lower()


    print(f"[LOADED] {os.path.basename(ds_path)} ({len(df_ds)} rows)")
    print(f"[LOADED] {os.path.basename(mat_path)} ({len(df_mat)} regime cells)")
    print(f"[LOADED] {os.path.basename(ex_path)} ({len(df_ex)} global zone examples)")
    return df_ds, df_mat, df_ex, ds_path




def select_matched_zone_example(
    target_lat: float,
    target_lon: float,
    target_aez_code: str,
    df_examples: pd.DataFrame,
    preferred_row_num: int | None = None,
) -> dict:
    """
    Matches the initial coordinate's resolved AEZXX code against 12_ex.csv and returns
    (LAT_ex, LON_ex) + metadata for passing into the main extraction pipeline.
    """
    df = df_examples.copy()


    # Resolve column names flexibly
    lat_col = next((c for c in df.columns if "lat" in c.lower()), "Latitude (--lat)")
    lon_col = next((c for c in df.columns if "lon" in c.lower()), "Longitude (--lon)")
    id_col = "#" if "#" in df.columns else df.columns[0]
    country_col = next((c for c in df.columns if "country" in c.lower()), "Country")
    city_col = next((c for c in df.columns if "city" in c.lower() or "basin" in c.lower()), "Agricultural City / Basin")
    prof_col = next((c for c in df.columns if "profile" in c.lower() or "output" in c.lower()), "Expected testb.py Output Profile")


    df["parsed_aez"] = df[prof_col].astype(str).apply(
        lambda s: (re.search(r"(AEZ\d{2})", s.upper()).group(1) if re.search(r"(AEZ\d{2})", s.upper()) else "AEZ01")
    )


    if preferred_row_num is not None:
        sub = df[df[id_col] == preferred_row_num]
        if not sub.empty:
            chosen = sub.iloc[0]
            return {
                "ex_id": int(chosen[id_col]),
                "country": str(chosen[country_col]).strip(),
                "city_basin": str(chosen[city_col]).strip(),
                "lat_ex": float(chosen[lat_col]),
                "lon_ex": float(chosen[lon_col]),
                "matched_aez": str(chosen["parsed_aez"]),
                "selection_mode": f"Manual Row #{preferred_row_num}",
            }


    matched = df[df["parsed_aez"] == target_aez_code].copy()
    if matched.empty:
        matched = df.copy()
        mode = f"Fallback Peer Example (no {target_aez_code} entry in 12_ex.csv)"
    else:
        mode = f"Matched AEZ Zone ({target_aez_code})"


    # Exclude identical coordinate (< ~1.5 km) when other matched-zone examples exist
    dist_deg = np.sqrt((matched[lat_col] - target_lat) ** 2 + (matched[lon_col] - target_lon) ** 2)
    non_identical = matched[dist_deg > 0.015]
    if not non_identical.empty:
        matched = non_identical


    lat_diff = matched[lat_col] - target_lat
    lon_diff = (matched[lon_col] - target_lon) * math.cos(math.radians(target_lat))
    matched["geo_dist_km"] = np.sqrt(lat_diff ** 2 + lon_diff ** 2) * 111.32
    chosen = matched.sort_values("geo_dist_km").iloc[0]


    return {
        "ex_id": int(chosen[id_col]),
        "country": str(chosen[country_col]).strip(),
        "city_basin": str(chosen[city_col]).strip(),
        "lat_ex": float(chosen[lat_col]),
        "lon_ex": float(chosen[lon_col]),
        "matched_aez": str(chosen["parsed_aez"]),
        "geo_dist_km": round(float(chosen["geo_dist_km"]), 1),
        "selection_mode": mode,
    }




# ==============================================================================
# SECTION 1: STATIC BASELINE LAYER (FAO HWSD v2.0 SOIL + NPK + pH, SRTM, AEZ01-10)
# ==============================================================================
def derive_npk_and_ph(
    clay_pct: float,
    sand_pct: float,
    total_n_g_kg: float | None,
    ph_water: float | None,
    org_carbon_g_kg: float | None,
    teb_cmol_kg: float | None,
) -> dict:
    if ph_water is not None and 3.0 <= ph_water <= 11.0:
        ph_val = round(float(ph_water), 2)
    else:
        ph_val = round(7.4 if clay_pct >= 35.0 else (6.5 if sand_pct >= 60.0 else 6.9), 2)


    n_base_g_kg = float(total_n_g_kg) if (total_n_g_kg is not None and total_n_g_kg > 0) else 1.25
    n_ratio = round(float(np.clip(n_base_g_kg * 38.0, 0.0, 140.0)), 1)


    oc_val = float(org_carbon_g_kg) if (org_carbon_g_kg is not None and org_carbon_g_kg > 0) else (8.0 + 0.15 * clay_pct)
    ph_availability_factor = max(0.45, 1.0 - 0.14 * abs(ph_val - 6.7))
    p_ratio = round(float(np.clip(oc_val * 3.4 * ph_availability_factor, 5.0, 145.0)), 1)


    teb_val = float(teb_cmol_kg) if (teb_cmol_kg is not None and teb_cmol_kg > 0) else (6.0 + 0.35 * clay_pct)
    k_ratio = round(float(np.clip(teb_val * 2.6 + clay_pct * 0.45, 5.0, 205.0)), 1)


    return {
        "ph": ph_val,
        "N_ratio": n_ratio,
        "P_ratio": p_ratio,
        "K_ratio": k_ratio,
        "baseline_nitrogen_g_kg": round(n_base_g_kg, 2),
    }




def query_local_hwsd2(lat: float, lon: float) -> dict | None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bil_path = os.path.join(base_dir, "HWSD2.bil")
    sqlite_path = os.path.join(base_dir, "HWSD2.sqlite")
    csv_path = os.path.join(base_dir, "HWSD2_LAYERS.csv")


    if not os.path.exists(bil_path):
        return None


    try:
        import rasterio
    except ImportError:
        print("      [WARN] HWSD2.bil found, but 'rasterio' is not installed. Run: pip install rasterio")
        return None


    t0 = time.time()
    with rasterio.open(bil_path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise RuntimeError(f"Coordinates ({lat}, {lon}) are outside HWSD2.bil raster bounds.")


        band = src.read(1)
        nodata = src.nodata if src.nodata is not None else 0
        smu_id = int(band[row, col])


        if smu_id == nodata or smu_id <= 0 or smu_id >= 65530:
            r_min, r_max = max(0, row - 1), min(src.height, row + 2)
            c_min, c_max = max(0, col - 1), min(src.width, col + 2)
            patch = band[r_min:r_max, c_min:c_max]
            valid_ids = patch[(patch != nodata) & (patch > 0) & (patch < 65530)]
            if len(valid_ids) == 0:
                raise RuntimeError(f"HWSD2 pixel ({lat}, {lon}) is water/ice/nodata (SMU_ID={smu_id}).")
            smu_id = int(valid_ids[0])


    clay_pct = sand_pct = silt_pct = None
    total_n = ph_water = org_carbon = teb_val = None
    wrb_class = f"HWSD2 SMU #{smu_id}"


    if os.path.exists(sqlite_path):
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(HWSD2_LAYERS)")
        avail_cols = {info[1].upper() for info in cur.fetchall()}


        wanted = ["CLAY", "SAND", "SILT", "TOTAL_N", "WRB4", "PH_WATER", "ORG_CARBON", "TEB"]
        select_cols = [c for c in wanted if c in avail_cols]


        cur.execute(
            f"""
            SELECT {', '.join(select_cols)}
            FROM HWSD2_LAYERS
            WHERE HWSD2_SMU_ID = ? AND LAYER = 'D1'
            ORDER BY SHARE DESC
            LIMIT 1
            """,
            (smu_id,),
        )
        row_data = cur.fetchone()
        if row_data:
            rec = dict(zip(select_cols, row_data))
            if rec.get("CLAY") is not None:
                clay_pct = round(float(rec["CLAY"]), 1)
                sand_pct = round(float(rec["SAND"]), 1)
                silt_pct = round(float(rec["SILT"]), 1)
                total_n = rec.get("TOTAL_N")
                ph_water = rec.get("PH_WATER")
                org_carbon = rec.get("ORG_CARBON")
                teb_val = rec.get("TEB")
                if rec.get("WRB4"):
                    wrb_class = f"{rec['WRB4']} (SMU #{smu_id})"


        if wrb_class.startswith("HWSD2 SMU"):
            try:
                cur.execute("SELECT WRB4 FROM HWSD2_SMU WHERE HWSD2_SMU_ID = ? LIMIT 1", (smu_id,))
                smu_row = cur.fetchone()
                if smu_row and smu_row[0]:
                    wrb_class = f"{smu_row[0]} (SMU #{smu_id})"
            except sqlite3.Error:
                pass
        conn.close()


    elif os.path.exists(csv_path):
        df_layers = pd.read_csv(csv_path)
        sub = df_layers[(df_layers["HWSD2_SMU_ID"] == smu_id) & (df_layers["LAYER"] == "D1")]
        if not sub.empty:
            top = sub.sort_values(by="SHARE", ascending=False).iloc[0]
            clay_pct = round(float(top["CLAY"]), 1)
            sand_pct = round(float(top["SAND"]), 1)
            silt_pct = round(float(top["SILT"]), 1)
            total_n = top.get("TOTAL_N")
            ph_water = top.get("PH_WATER")
            org_carbon = top.get("ORG_CARBON")
            teb_val = top.get("TEB")
            if "WRB4" in top and pd.notna(top["WRB4"]):
                wrb_class = f"{top['WRB4']} (SMU #{smu_id})"
    else:
        raise RuntimeError(
            f"HWSD2.bil resolved SMU_ID={smu_id}, but neither 'HWSD2.sqlite' nor 'HWSD2_LAYERS.csv' "
            f"was found in {base_dir}."
        )


    if clay_pct is None or sand_pct is None:
        raise RuntimeError(f"SMU_ID={smu_id} found in HWSD2.bil, but D1 soil properties are null/water.")


    chem = derive_npk_and_ph(clay_pct, sand_pct, total_n, ph_water, org_carbon, teb_val)
    elapsed = round(time.time() - t0, 3)
    print(
        f"      [OK] Local FAO HWSD v2.0 ({elapsed}s) -> {wrb_class} | "
        f"Clay: {clay_pct}%, Sand: {sand_pct}%, pH: {chem['ph']}"
    )
    return {
        "wrb_class": wrb_class,
        "clay_pct": clay_pct,
        "sand_pct": sand_pct,
        "silt_pct": silt_pct,
        "ph": chem["ph"],
        "N_ratio": chem["N_ratio"],
        "P_ratio": chem["P_ratio"],
        "K_ratio": chem["K_ratio"],
        "baseline_nitrogen_g_kg": chem["baseline_nitrogen_g_kg"],
        "source": "FAO HWSD v2.0 (Local Raster + DB)",
    }




def get_static_soil(lat: float, lon: float) -> dict:
    print(f"  [SECTION 1A] Resolving Static Soil Chemistry (NPK, pH, Texture) for ({lat}, {lon})...")
    try:
        local_soil = query_local_hwsd2(lat, lon)
        if local_soil is not None:
            return local_soil
    except Exception as e:
        print(f"      [HWSD2 NOTICE] {e}")


    t0 = time.time()
    soil_endpoints = [
        ("PixelGust SoilGrids API", "https://pixelgust.com/soilgrids/v2.0/properties/query"),
        ("ISRIC SoilGrids Official API", "https://rest.isric.org/soilgrids/v2.0/properties/query"),
    ]
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["clay", "sand", "silt", "nitrogen", "phh2o", "soc"],
        "depth": "0-5cm",
        "value": "mean",
    }


    errors = []
    for provider, url in soil_endpoints:
        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code == 429:
                errors.append(f"{provider}: Rate limited (HTTP 429)")
                continue
            res.raise_for_status()


            layers = res.json().get("properties", {}).get("layers", [])
            metrics = {l["name"]: l["depths"][0]["values"]["mean"] for l in layers}


            if metrics.get("clay") is None or metrics.get("sand") is None:
                errors.append(f"{provider}: Pixel ({lat}, {lon}) masked (water/urban/ice).")
                continue


            clay_pct = round(metrics["clay"] / 10.0, 1)
            sand_pct = round(metrics["sand"] / 10.0, 1)
            silt_pct = round((metrics.get("silt") or 0) / 10.0, 1)
            total_n = (metrics.get("nitrogen") or 125) / 100.0
            ph_water = (metrics.get("phh2o") / 10.0) if metrics.get("phh2o") else None
            org_carbon = (metrics.get("soc") / 10.0) if metrics.get("soc") else None


            chem = derive_npk_and_ph(clay_pct, sand_pct, total_n, ph_water, org_carbon, None)
            print(f"      [OK] {provider} ({round(time.time() - t0, 2)}s) -> Clay: {clay_pct}%, pH: {chem['ph']}")
            return {
                "wrb_class": "SoilGrids WRB",
                "clay_pct": clay_pct,
                "sand_pct": sand_pct,
                "silt_pct": silt_pct,
                "ph": chem["ph"],
                "N_ratio": chem["N_ratio"],
                "P_ratio": chem["P_ratio"],
                "K_ratio": chem["K_ratio"],
                "baseline_nitrogen_g_kg": chem["baseline_nitrogen_g_kg"],
                "source": provider,
            }
        except Exception as e:
            errors.append(f"{provider}: {e}")


    print("\n[ERROR - SECTION 1A SOIL] Could not retrieve soil data from local HWSD2 or online APIs:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)




def get_static_terrain(lat: float, lon: float) -> dict:
    print("  [SECTION 1B] Fetching NASA SRTM Elevation & Computing Terrain Slope...")
    t0 = time.time()
    delta = 0.001
    locations = (
        f"{lat},{lon}|{lat + delta},{lon}|{lat - delta},{lon}|{lat},{lon + delta}|{lat},{lon - delta}"
    )
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={locations}"
    try:
        res = requests.get(url, timeout=12)
        res.raise_for_status()
        elevs = [r.get("elevation") for r in res.json().get("results", [])]
        if len(elevs) < 5 or any(e is None for e in elevs):
            raise RuntimeError("Null elevation returned.")


        z_c, z_n, z_s, z_e, z_w = elevs
        dy_m = 2.0 * delta * 111320.0
        dx_m = 2.0 * delta * 111320.0 * max(0.1, math.cos(math.radians(lat)))
        slope_pct = round(math.sqrt(((z_e - z_w) / dx_m) ** 2 + ((z_n - z_s) / dy_m) ** 2) * 100.0, 2)
        erosion_risk = "Low Runoff" if slope_pct < 2.0 else ("Moderate Runoff" if slope_pct < 6.0 else "High Erosion Risk")
        print(f"      [OK] NASA SRTM 30m ({round(time.time() - t0, 2)}s) -> Elev: {z_c:.1f} m | Slope: {slope_pct}% ({erosion_risk})")
        return {"elevation_m": round(float(z_c), 1), "slope_pct": slope_pct, "erosion_risk": erosion_risk}
    except Exception as e:
        print(f"      [SRTM NOTICE] Terrain feed skipped ({e})")
        return {"elevation_m": None, "slope_pct": None, "erosion_risk": "Unavailable"}




def get_static_climate_and_aez(lat: float, lon: float, soil_info: dict) -> dict:
    print("  [SECTION 1C] Fetching NASA Climatology & Deriving AEZ01-AEZ10 Prefix...")
    t0 = time.time()
    url = (
        f"https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&format=JSON"
    )
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        payload = res.json()
        if "properties" not in payload:
            raise RuntimeError(f"Unexpected NASA response: {payload}")
    except Exception as e:
        print(f"\n[API ERROR - SECTION 1C CLIMATOLOGY] NASA POWER Climatology failed: {e}")
        sys.exit(1)


    data = payload["properties"]["parameter"]
    annual_temp = data["T2M"]["ANN"]
    annual_precip_mm = data["PRECTOTCORR"]["ANN"] * 365.0
    monthly_temps = [v for k, v in data["T2M"].items() if k != "ANN"]
    min_month_temp = min(monthly_temps)
    max_month_temp = max(monthly_temps)


    if max_month_temp < 10.0:
        aez_code = "AEZ06"
        climate_zone = "Polar / Subarctic Tundra (ET - Cold Limit)"
        aez_label = "AEZ06 (Cold Continental / Boreal Limit)"
    elif min_month_temp < 0.0:
        aez_code = "AEZ06"
        climate_zone = "Temperate / Continental (Dfa/Dfb)"
        aez_label = "AEZ06 (Temperate Continental Prairie - Dfa/Dfb)"
    elif annual_precip_mm < 250.0:
        climate_zone = "Hot Arid (BWh)"
        if "fluvisol" in soil_info["wrb_class"].lower() or soil_info["clay_pct"] >= 30.0:
            aez_code = "AEZ01"
            aez_label = "AEZ01 (Hot Arid Alluvial Valley - BWh)"
        else:
            aez_code = "AEZ02"
            aez_label = "AEZ02 (Hot Arid Desert / Steppe - BWh/BWk)"
    elif annual_precip_mm < 500.0:
        aez_code = "AEZ04"
        climate_zone = "Semi-Arid Subtropical Steppe (BSh)"
        aez_label = "AEZ04 (Semi-Arid Subtropical Loam - BSh)"
    elif annual_precip_mm > 1800.0 and min_month_temp >= 22.0:
        aez_code = "AEZ10"
        climate_zone = "Equatorial Humid Rainforest (Af)"
        aez_label = "AEZ10 (Equatorial Humid Rainforest - Af)"
    elif annual_precip_mm > 1400.0 and min_month_temp >= 18.0:
        aez_code = "AEZ09"
        climate_zone = "Tropical Monsoon Delta (Am)"
        aez_label = "AEZ09 (Tropical Monsoon Delta - Am)"
    elif min_month_temp >= 18.0 and annual_temp < 23.5:
        aez_code = "AEZ08"
        climate_zone = "Tropical Highland Volcanic (Cwb/Cfb)"
        aez_label = "AEZ08 (Tropical Highland Volcanic - Cwb)"
    elif min_month_temp >= 18.0:
        aez_code = "AEZ07"
        climate_zone = "Tropical Wet-Dry Savanna (Aw)"
        aez_label = "AEZ07 (Tropical Wet-Dry Savanna - Aw)"
    elif annual_precip_mm >= 900.0:
        aez_code = "AEZ05"
        climate_zone = "Humid Subtropical Plain (Cfa/Cwa)"
        aez_label = "AEZ05 (Humid Subtropical Plain - Cfa/Cwa)"
    else:
        aez_code = "AEZ03"
        climate_zone = "Mediterranean Warm-Temperate Alluvial (Csa)"
        aez_label = "AEZ03 (Mediterranean Warm-Temperate Alluvial - Csa)"


    print(f"      [OK] NASA Climatology ({round(time.time() - t0, 2)}s) -> {climate_zone} | Prefix: {aez_code}")
    return {
        "aez_code": aez_code,
        "climate_zone": climate_zone,
        "annual_temp_c": round(annual_temp, 1),
        "annual_precip_mm": round(annual_precip_mm, 1),
        "aez_profile": aez_label,
    }




# ==============================================================================
# SECTION 2: DYNAMIC TELEMETRY LAYER (TEMP, HUMIDITY, H2O, N, MODIS NDVI)
# ==============================================================================
def fetch_modis_ndvi_channel(lat: float, lon: float) -> pd.DataFrame | None:
    print("      [2D - CANOPY] Fetching NASA MODIS (MOD13Q1 250m) NDVI Vegetation Vigor...")
    t0 = time.time()
    try:
        dates_url = f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/dates?latitude={lat}&longitude={lon}"
        d_res = requests.get(dates_url, timeout=12)
        d_res.raise_for_status()
        dates_list = d_res.json().get("dates", [])
        if len(dates_list) < 4:
            raise RuntimeError("Insufficient MODIS dates.")


        start_modis = dates_list[-8]["modis_date"]
        end_modis = dates_list[-1]["modis_date"]
        sub_url = (
            f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
            f"?latitude={lat}&longitude={lon}&band=250m_16_days_NDVI"
            f"&startDate={start_modis}&endDate={end_modis}&kmAboveBelow=0&kmLeftRight=0"
        )
        s_res = requests.get(sub_url, timeout=15)
        s_res.raise_for_status()
        subsets = s_res.json().get("subset", [])


        records = []
        for item in subsets:
            val = item["data"][0]
            if val > -2000:
                records.append({"date": pd.to_datetime(item["calendar_date"]), "ndvi": round(val * 0.0001, 3)})


        if not records:
            raise RuntimeError("Cloud-masked NDVI composites.")


        df_ndvi = pd.DataFrame(records).set_index("date").sort_index()
        print(f"      [OK] NASA MODIS NDVI ({round(time.time() - t0, 2)}s) -> Latest NDVI: {df_ndvi['ndvi'].iloc[-1]:.3f}")
        return df_ndvi
    except Exception as e:
        print(f"      [MODIS NOTICE] NDVI channel skipped ({e})")
        return None




def get_dynamic_telemetry(lat: float, lon: float, soil_info: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    print("  [SECTION 2] Ingesting Dynamic Telemetry Channels (Temp, Humidity, Rainfall, H2O, N)...")
    t0 = time.time()


    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=45)
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")


    nasa_url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,WS2M,GWETROOT,GWETTOP"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start_str}&end={end_str}&format=JSON"
    )
    try:
        res = requests.get(nasa_url, timeout=20)
        res.raise_for_status()
        raw_resp = res.json()
        if "properties" not in raw_resp:
            raise RuntimeError(f"NASA POWER API Error: {raw_resp}")
    except Exception as e:
        print(f"\n[API ERROR - SECTION 2 NASA TELEMETRY] Failed to retrieve satellite telemetry: {e}")
        sys.exit(1)


    df_hist = pd.DataFrame(raw_resp["properties"]["parameter"])
    df_hist.index = pd.to_datetime(df_hist.index, format="%Y%m%d")
    df_hist = df_hist.replace(-999.0, np.nan)
    df_hist = df_hist.dropna(subset=["T2M", "GWETROOT"], how="any")
    df_hist = df_hist.interpolate(method="linear").bfill().ffill()
    print(
        f"      [2A/2B - TEMP, HUMIDITY & SMAP H2O] Loaded {len(df_hist)} days "
        f"(latest pass: {df_hist.index[-1].strftime('%Y-%m-%d')}) in {round(time.time() - t0, 2)}s"
    )


    df_hist["ET0_est_mm"] = (
        0.0023
        * (df_hist["T2M"] + 17.8)
        * np.sqrt(np.maximum(0.5, df_hist["T2M_MAX"] - df_hist["T2M_MIN"]))
        * (df_hist["ALLSKY_SFC_SW_DWN"] * 0.408)
    ).clip(lower=0.5)
    df_hist["H2O_balance_mm"] = df_hist["PRECTOTCORR"] - df_hist["ET0_est_mm"]


    n0 = soil_info["baseline_nitrogen_g_kg"] if soil_info["baseline_nitrogen_g_kg"] > 0 else 1.25
    sand_frac = soil_info["sand_pct"] / 100.0
    infiltration_cap_mm = 18.0 * (1.0 - 0.5 * sand_frac)


    n_series = []
    current_n = n0
    for _, row in df_hist.iterrows():
        excess_rain = max(0.0, row["PRECTOTCORR"] - infiltration_cap_mm)
        leach_loss = current_n * (0.012 * (0.4 + sand_frac)) * (excess_rain / 10.0)
        mineralization_gain = (0.0015 * n0) if (18.0 <= row["T2M"] <= 32.0 and 0.35 <= row["GWETROOT"] <= 0.80) else 0.0
        current_n = max(0.10, current_n - leach_loss + mineralization_gain)
        n_series.append(round(current_n, 4))


    df_hist["N2_dynamic_g_kg"] = n_series
    df_hist["N_dynamic_ratio"] = (df_hist["N2_dynamic_g_kg"] * 38.0).clip(lower=0.0, upper=140.0)
    print(
        f"      [2C - DYNAMIC N] Coupled mass-balance computed (N0: {n0} g/kg -> "
        f"Current: {df_hist['N2_dynamic_g_kg'].iloc[-1]:.3f} g/kg | N-Ratio: {df_hist['N_dynamic_ratio'].iloc[-1]:.1f})"
    )


    df_ndvi = fetch_modis_ndvi_channel(lat, lon)


    print("      [2E - FORECAST] Fetching 7-Day Weather, Wind & ET0 Forecast...")
    t1 = time.time()
    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,et0_fao_evapotranspiration,wind_speed_10m_max"
        f"&hourly=soil_moisture_3_to_9cm&forecast_days=7&timezone=auto"
    )
    try:
        fc_res = requests.get(forecast_url, timeout=20)
        fc_res.raise_for_status()
        fc_json = fc_res.json()
        if "daily" not in fc_json:
            raise RuntimeError(f"Open-Meteo unexpected response: {fc_json}")
    except Exception as e:
        print(f"\n[API ERROR - SECTION 2E FORECAST] Open-Meteo Forecast failed: {e}")
        sys.exit(1)


    hourly_df = pd.DataFrame({
        "time": pd.to_datetime(fc_json["hourly"]["time"]),
        "fc_soil_moisture": fc_json["hourly"]["soil_moisture_3_to_9cm"],
    })
    daily_sm = hourly_df.resample("D", on="time")["fc_soil_moisture"].mean()


    df_fc = pd.DataFrame({
        "date": pd.to_datetime(fc_json["daily"]["time"]),
        "fc_temp_c": fc_json["daily"]["temperature_2m_mean"],
        "fc_temp_max_c": fc_json["daily"]["temperature_2m_max"],
        "fc_temp_min_c": fc_json["daily"]["temperature_2m_min"],
        "fc_precip_mm": fc_json["daily"]["precipitation_sum"],
        "fc_et0_mm": fc_json["daily"]["et0_fao_evapotranspiration"],
        "fc_wind_kmh": fc_json["daily"]["wind_speed_10m_max"],
    }).set_index("date")
    df_fc["fc_soil_moisture"] = daily_sm
    print(f"      [OK] 7-Day Forecast ({round(time.time() - t1, 2)}s)")


    return df_hist, df_fc, df_ndvi




# ==============================================================================
# SECTION 3: DSP SIGNAL CONDITIONING (SAVITZKY-GOLAY LOW-PASS FIR FILTER)
# ==============================================================================
def apply_dsp_conditioning(df_hist: pd.DataFrame) -> pd.DataFrame:
    print("  [SECTION 3] Applying Savitzky-Golay DSP Filter (Temp, H2O, N)...")
    if len(df_hist) >= 5:
        win = min(7, len(df_hist) if len(df_hist) % 2 == 1 else len(df_hist) - 1)
        df_hist["T2M_filtered"] = savgol_filter(df_hist["T2M"], window_length=win, polyorder=2)
        df_hist["GWETROOT_filtered"] = savgol_filter(df_hist["GWETROOT"], window_length=win, polyorder=2)
        df_hist["N2_filtered"] = savgol_filter(df_hist["N2_dynamic_g_kg"], window_length=win, polyorder=2)
        df_hist["N_ratio_filtered"] = savgol_filter(df_hist["N_dynamic_ratio"], window_length=win, polyorder=2)
    else:
        df_hist["T2M_filtered"] = df_hist["T2M"]
        df_hist["GWETROOT_filtered"] = df_hist["GWETROOT"]
        df_hist["N2_filtered"] = df_hist["N2_dynamic_g_kg"]
        df_hist["N_ratio_filtered"] = df_hist["N_dynamic_ratio"]
    print("      [OK] High-frequency orbital noise attenuated across primary channels.")
    return df_hist




# ==============================================================================
# SECTION 4: TIER 1 SIMPLE | TIER 2 ROTATION (2 PLANS) | TIER 3 AEZXX x MJO
# ==============================================================================
HEAVY_FEEDER_CROPS = {"maize", "cotton", "rice", "banana", "watermelon", "muskmelon", "jute", "coffee"}




def fetch_live_mjo_phase() -> tuple[str, float, str]:
    url = "https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        lines = [ln.strip() for ln in res.text.splitlines() if ln.strip() and not ln.startswith("#")]
        parts = re.split(r"\s+", lines[-1])
        obs_date = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        pc1, pc2, amp = float(parts[4]), float(parts[5]), float(parts[6])
        if amp < 1.0:
            return "MJO-P0", round(amp, 2), f"NOAA PSL OMI ({obs_date})"
        angle_deg = (np.degrees(np.arctan2(-pc2, -pc1)) + 360.0) % 360.0
        phase_num = int(angle_deg // 45.0) + 1
        return f"MJO-P{phase_num}", round(amp, 2), f"NOAA PSL OMI ({obs_date})"
    except Exception:
        return "MJO-P0", 0.0, "Offline Climatological Fallback"




def run_tier1_simple_decisions(
    df_hist: pd.DataFrame,
    df_fc: pd.DataFrame,
    features_7: dict,
    soil_info: dict,
    terrain_info: dict,
    df_crop_ext: pd.DataFrame,
    previous_crop: str = "none",
) -> dict:
    root_h2o = float(df_hist["GWETROOT_filtered"].iloc[-1])
    temp_c = float(features_7["temperature"])
    rh_pct = float(features_7["humidity"])
    n_val, p_val, k_val = features_7["N"], features_7["P"], features_7["K"]


    rain_7d = float(df_fc["fc_precip_mm"].sum())
    et0_7d = float(df_fc["fc_et0_mm"].sum())
    water_deficit_7d = round(et0_7d - rain_7d, 1)
    max_wind_kmh = float(df_fc["fc_wind_kmh"].max())
    max_temp_7d = float(df_fc["fc_temp_max_c"].max())
    slope_pct = terrain_info["slope_pct"] if terrain_info["slope_pct"] is not None else 1.0


    # 1. IRRIGATION (HIGH / MID / LOW)
    if root_h2o < 0.42 and rain_7d < 5.0:
        irr_level = "HIGH"
        irr_action = (
            f"Root H2O low ({root_h2o:.2f}) with {water_deficit_7d} mm weekly water deficit. "
            f"Run full evening irrigation cycle within 24h."
        )
    elif (0.42 <= root_h2o <= 0.62 and rain_7d < 12.0) or (12.0 <= water_deficit_7d <= 30.0 and root_h2o < 0.65):
        irr_level = "MID"
        irr_action = (
            f"Root H2O moderate ({root_h2o:.2f}). Apply 50% maintenance drip pulse "
            f"to cover {et0_7d:.1f} mm weekly ET0 demand."
        )
    else:
        irr_level = "LOW"
        irr_action = (
            f"Hold irrigation: Root H2O sufficient ({root_h2o:.2f}) or incoming rain "
            f"({rain_7d:.1f} mm) satisfies crop demand."
        )


    # 2. COVER CROPS & WEATHER PROTECTION (HIGH / MID / LOW)
    if (rain_7d > 25.0 and slope_pct >= 3.0) or max_wind_kmh > 38.0:
        cover_level = "HIGH"
        cover_action = (
            f"Heavy rain ({rain_7d:.1f} mm on {slope_pct}% slope) or gale wind ({max_wind_kmh:.1f} km/h). "
            f"Maintain dense cover crop (Berseem Clover / Rye) + perimeter windbreak."
        )
    elif (12.0 <= rain_7d <= 25.0) or (25.0 <= max_wind_kmh <= 38.0) or max_temp_7d > 35.0:
        cover_level = "MID"
        cover_action = (
            f"Moderate wind ({max_wind_kmh:.1f} km/h) or thermal load ({max_temp_7d:.1f} C max). "
            f"Apply inter-row organic mulch to prevent erosion and evaporation."
        )
    else:
        cover_level = "LOW"
        cover_action = (
            f"Calm wind ({max_wind_kmh:.1f} km/h), low rain ({rain_7d:.1f} mm), gentle slope ({slope_pct}%). "
            f"Standard open-field soil surface management."
        )


    # 3. NUTRITION REPLENISHMENT AFTER HEAVY CROP (HIGH / MID / LOW)
    n_p35, n_p65 = float(df_crop_ext["N"].quantile(0.35)), float(df_crop_ext["N"].quantile(0.65))
    p_p35, p_p65 = float(df_crop_ext["P"].quantile(0.35)), float(df_crop_ext["P"].quantile(0.65))
    k_p35 = float(df_crop_ext["K"].quantile(0.35))


    prev_clean = previous_crop.strip().lower()
    was_heavy_crop = prev_clean in HEAVY_FEEDER_CROPS
    leach_note = " [Split dose: rain >20mm in sandy soil]" if (rain_7d > 20.0 and soil_info["sand_pct"] > 45.0) else ""


    if was_heavy_crop or n_val < n_p35 or p_val < p_p35 or k_val < k_p35:
        nutr_level = "HIGH"
        trigger_msg = f"post-heavy crop ({prev_clean})" if was_heavy_crop else "below dataset 35th percentile"
        nutr_action = (
            f"Replenishment required ({trigger_msg}): NPK=({n_val}/{p_val}/{k_val}). "
            f"Apply balanced NPK + compost recovery dose{leach_note}."
        )
    elif n_val < n_p65 or p_val < p_p65:
        nutr_level = "MID"
        nutr_action = (
            f"Moderate reserves (prev: {prev_clean}): NPK=({n_val}/{p_val}/{k_val}). "
            f"Apply light maintenance fertigation{leach_note}."
        )
    else:
        nutr_level = "LOW"
        nutr_action = f"High soil reserves: NPK=({n_val}/{p_val}/{k_val}); hold synthetic fertilizer."


    # 4. INSECTICIDE & BIO-SPRAY MANAGEMENT (HIGH / MID / LOW)
    wind_gate = " [DELAY SPRAY UNTIL CALM DAWN: Wind >25 km/h drift risk]" if max_wind_kmh > 25.0 else ""
    if 20.0 <= temp_c <= 33.0 and rh_pct > 72.0:
        pest_level = "HIGH"
        pest_action = (
            f"Warm + humid ({temp_c:.1f} C, {rh_pct:.0f}% RH) triggers larvae/fungal outbreak. "
            f"Apply targeted bio-insecticide within 48h{wind_gate}."
        )
    elif temp_c > 33.0 and rh_pct < 30.0:
        pest_level = "HIGH"
        pest_action = (
            f"Hot-dry stress ({temp_c:.1f} C, {rh_pct:.0f}% RH) favors spider mites/thrips. "
            f"Apply evening miticide / horticultural oil{wind_gate}."
        )
    elif 20.0 <= temp_c <= 33.0 and (62.0 <= rh_pct <= 72.0):
        pest_level = "MID"
        pest_action = (
            f"Moderate humidity ({rh_pct:.0f}% RH). Deploy sticky/pheromone traps "
            f"and preventative neem oil spray{wind_gate}."
        )
    else:
        pest_level = "LOW"
        pest_action = f"Low pest pressure ({temp_c:.1f} C, {rh_pct:.0f}% RH); routine scouting only."


    return {
        "irrigation":  {"level": irr_level,   "action": irr_action},
        "cover_crop":  {"level": cover_level, "action": cover_action},
        "nutrition":   {"level": nutr_level,  "action": nutr_action},
        "insecticide": {"level": pest_level,  "action": pest_action},
    }




def run_tier2_two_rotation_plans(
    df_crop_ext: pd.DataFrame,
    features_7: dict,
    effective_water_mm: float,
    tier1: dict,
    previous_crop: str = "none",
    num_seasons: int = 3,
) -> dict:
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(df_crop_ext[feature_cols], df_crop_ext["label"])


    stats = df_crop_ext.groupby("label")[feature_cols].mean()
    n_q35, n_q65 = float(df_crop_ext["N"].quantile(0.35)), float(df_crop_ext["N"].quantile(0.65))
    rain_q35, rain_q65 = float(df_crop_ext["rainfall"].quantile(0.35)), float(df_crop_ext["rainfall"].quantile(0.65))
    n_max, p_max, k_max = (
        float(df_crop_ext["N"].max()),
        float(df_crop_ext["P"].max()),
        float(df_crop_ext["K"].max()),
    )


    crop_meta = {}
    for crop, row in stats.iterrows():
        if row["N"] <= n_q35:
            role = "Low-N / Restorative"
            dN = +18.0
        elif row["N"] >= n_q65:
            role = "Heavy-N Feeder"
            dN = -28.0
        else:
            role = "Moderate Feeder"
            dN = -12.0


        dP = -round(float(row["P"]) * 0.14, 1)
        dK = -round(float(row["K"]) * 0.14, 1)


        if row["rainfall"] <= rain_q35:
            water_tier = "LOW"
        elif row["rainfall"] >= rain_q65:
            water_tier = "HIGH"
        else:
            water_tier = "MID"


        crop_meta[crop] = {
            "role": role,
            "dN": dN,
            "dP": dP,
            "dK": dK,
            "water_tier": water_tier,
            "mean_N": round(float(row["N"]), 1),
        }


    base_state = {
        "N": float(features_7["N"]),
        "P": float(features_7["P"]),
        "K": float(features_7["K"]),
        "temperature": float(features_7["temperature"]),
        "humidity": float(features_7["humidity"]),
        "ph": float(features_7["ph"]),
        "rainfall": float(effective_water_mm),
    }
    nutr_level = tier1["nutrition"]["level"]
    if nutr_level == "HIGH":
        base_state["N"] = min(n_max, base_state["N"] + 18.0)
        base_state["P"] = min(p_max, base_state["P"] + 12.0)
        base_state["K"] = min(k_max, base_state["K"] + 12.0)
    elif nutr_level == "MID":
        base_state["N"] = min(n_max, base_state["N"] + 8.0)
        base_state["P"] = min(p_max, base_state["P"] + 5.0)
        base_state["K"] = min(k_max, base_state["K"] + 5.0)


    irr_level = tier1["irrigation"]["level"]
    prev_clean = previous_crop.strip().lower()
    is_fallow_start = prev_clean in ("", "none", "fallow", "n/a")
    start_label = "none (Fallow)" if is_fallow_start else prev_clean


    def simulate_plan(strategy: str, blocked_by_season: dict[int, set[str]] | None = None) -> dict:
        state = dict(base_state)
        initial_n = state["N"]
        history = [] if is_fallow_start else [prev_clean]
        seasons_out = []
        blocked_map = blocked_by_season or {}


        for s_idx in range(1, num_seasons + 1):
            q_df = pd.DataFrame([state])[feature_cols]
            probs = clf.predict_proba(q_df)[0]
            class_probs = dict(zip(clf.classes_, probs))


            prev_role = (
                crop_meta[history[-1]]["role"]
                if (history and history[-1] in crop_meta)
                else "Fallow / Neutral"
            )
            best_crop = None
            best_score = -1.0
            best_prob = 0.0


            for crop, ml_prob in class_probs.items():
                if crop in history[-2:]:
                    continue
                if crop in blocked_map.get(s_idx, set()):
                    continue


                m = crop_meta[crop]
                multiplier = 1.0


                if strategy == "plan_a_restoration":
                    if m["role"] == "Low-N / Restorative":
                        multiplier += 1.20
                    elif m["role"] == "Moderate Feeder":
                        multiplier += 0.45
                    if m["water_tier"] == "LOW":
                        multiplier += 0.85 if irr_level in ("HIGH", "MID") else 0.35
                    elif m["water_tier"] == "HIGH":
                        multiplier *= 0.35
                    if prev_role == "Heavy-N Feeder" and m["role"] == "Low-N / Restorative":
                        multiplier += 0.60


                elif strategy == "plan_b_commercial":
                    if m["role"] == "Heavy-N Feeder" and state["N"] >= 50.0:
                        multiplier += 1.35
                    elif m["role"] == "Moderate Feeder":
                        multiplier += 0.90
                    elif prev_role == "Heavy-N Feeder" and m["role"] == "Low-N / Restorative":
                        multiplier += 0.85
                    if irr_level == "HIGH" and m["water_tier"] == "HIGH":
                        multiplier *= 0.50


                score = (ml_prob + 0.005) * multiplier
                if score > best_score:
                    best_score = score
                    best_crop = crop
                    best_prob = ml_prob


            chosen = crop_meta[best_crop]
            pre_npk = (round(state["N"], 1), round(state["P"], 1), round(state["K"], 1))
            state["N"] = float(np.clip(state["N"] + chosen["dN"], 5.0, n_max))
            state["P"] = float(np.clip(state["P"] + chosen["dP"], 5.0, p_max))
            state["K"] = float(np.clip(state["K"] + chosen["dK"], 5.0, k_max))
            post_npk = (round(state["N"], 1), round(state["P"], 1), round(state["K"], 1))


            seasons_out.append({
                "season": f"Season {s_idx}",
                "crop": best_crop,
                "role": chosen["role"],
                "water_demand": chosen["water_tier"],
                "ml_match_pct": round(best_prob * 100.0, 1),
                "pre_npk": pre_npk,
                "post_npk": post_npk,
            })
            history.append(best_crop)


        return {
            "sequence": [s["crop"] for s in seasons_out],
            "seasons": seasons_out,
            "net_N_delta": round(state["N"] - initial_n, 1),
        }


    plan_a = simulate_plan("plan_a_restoration")
    plan_b = simulate_plan(
        "plan_b_commercial",
        blocked_by_season={
            1: {plan_a["sequence"][0]},
            3: {plan_a["sequence"][2]},
        },
    )


    return {
        "starting_crop": start_label,
        "plan_a": {"title": "Plan A: Soil-Restoration & Water-Saving Cycle", **plan_a},
        "plan_b": {"title": "Plan B: Commercial Cash-Crop & Balanced-NPK Cycle", **plan_b},
    }




def run_tier3_aez_mjo_connection(
    aez_code: str,
    features_7: dict,
    effective_water_mm: float,
    df_aez_mjo: pd.DataFrame,
    df_matrix_90: pd.DataFrame,
    tier2: dict,
) -> dict:
    mjo_code, mjo_amp, mjo_source = fetch_live_mjo_phase()
    composite_key = f"{aez_code}_{mjo_code}"


    mat_sub = df_matrix_90[df_matrix_90["aez_mjo_key"] == composite_key]
    mat_row = mat_sub.iloc[0] if not mat_sub.empty else df_matrix_90[df_matrix_90["aez_code"] == aez_code].iloc[0]


    sub_exact = df_aez_mjo[df_aez_mjo["aez_mjo_key"] == composite_key]
    sub_aez = df_aez_mjo[df_aez_mjo["aez_code"] == aez_code]
    train_df = sub_exact if len(sub_exact) >= 25 else (sub_aez if len(sub_aez) >= 20 else df_aez_mjo)


    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(train_df[feature_cols], train_df["label"])


    mjo_rows = df_aez_mjo[df_aez_mjo["mjo_code"] == mjo_code]
    temp_offset = float(mjo_rows["mjo_temp_offset_c"].iloc[0]) if not mjo_rows.empty else 0.0
    rh_offset = float(mjo_rows["mjo_humidity_offset_pct"].iloc[0]) if not mjo_rows.empty else 0.0
    rain_mult = float(mat_row["mjo_rain_multiplier"])


    q_vec = pd.DataFrame([{
        "N": features_7["N"],
        "P": features_7["P"],
        "K": features_7["K"],
        "temperature": round(features_7["temperature"] + temp_offset, 2),
        "humidity": round(np.clip(features_7["humidity"] + rh_offset, 15.0, 98.0), 2),
        "ph": features_7["ph"],
        "rainfall": round(effective_water_mm * rain_mult, 1),
    }])
    probs = clf.predict_proba(q_vec)[0]
    mjo_top3 = [(str(clf.classes_[i]), round(float(probs[i]) * 100.0, 1)) for i in np.argsort(probs)[::-1][:3]]


    stats = sub_aez.groupby("label")[feature_cols].mean() if len(sub_aez) >= 10 else df_aez_mjo.groupby("label")[feature_cols].mean()
    rain_max_ds = float(df_aez_mjo["rainfall"].max())
    n_max_ds = float(df_aez_mjo["N"].max())


    def score_plan(plan_dict: dict) -> dict:
        wue_list, nue_list = [], []
        for c in plan_dict["sequence"]:
            if c in stats.index:
                wue_list.append(max(10.0, 100.0 * (1.0 - float(stats.loc[c, "rainfall"]) / rain_max_ds)))
                nue_list.append(max(10.0, 100.0 * (1.0 - float(stats.loc[c, "N"]) / n_max_ds)))
        wue = round(float(np.mean(wue_list)), 1) if wue_list else 55.0
        nue = round(float(np.mean(nue_list)), 1) if nue_list else 55.0
        return {
            "wue_pct": wue,
            "nue_pct": nue,
            "global_score": round(0.55 * wue + 0.45 * nue, 1),
        }


    return {
        "aez_code": aez_code,
        "mjo_code": mjo_code,
        "aez_mjo_key": composite_key,
        "prefix_pattern": f"{composite_key}_NNNN",
        "mjo_amplitude": mjo_amp,
        "mjo_source": mjo_source,
        "convective_region": str(mat_row["mjo_convective_region"]),
        "teleconnection_regime": str(mat_row["mjo_teleconnection_regime"]),
        "global_irrigation_modifier": str(mat_row["global_irrigation_modifier"]),
        "benchmark_basins": str(mat_row["global_benchmark_basins"]),
        "export_corridor": str(mat_row["global_export_corridor"]),
        "rain_multiplier": rain_mult,
        "temp_offset_c": temp_offset,
        "rh_offset_pct": rh_offset,
        "matched_slice_rows": int(len(sub_exact)),
        "aez_total_rows": int(len(sub_aez)),
        "matrix_top3_ref": str(mat_row["recommended_crops_top3"]),
        "mjo_adjusted_crops": mjo_top3,
        "plan_a_global": score_plan(tier2["plan_a"]),
        "plan_b_global": score_plan(tier2["plan_b"]),
    }




def run_decision_engine(
    df_hist: pd.DataFrame,
    df_fc: pd.DataFrame,
    df_ndvi: pd.DataFrame | None,
    static_aez: dict,
    soil_info: dict,
    terrain_info: dict,
    df_aez_mjo: pd.DataFrame,
    df_matrix_90: pd.DataFrame,
    dataset_path: str,
    previous_crop: str,
) -> dict:
    print("  [SECTION 4] Running Tier 1 Simple, Tier 2 Rotation (2 Plans) & Tier 3 AEZXX x MJO Engine...")


    latest_temp = float(df_hist["T2M_filtered"].iloc[-1])
    latest_humidity = float(df_hist["RH2M"].iloc[-1])
    latest_root_h2o = float(df_hist["GWETROOT_filtered"].iloc[-1])
    latest_n_ratio = round(float(df_hist["N_ratio_filtered"].iloc[-1]), 1)
    latest_ndvi = float(df_ndvi["ndvi"].iloc[-1]) if df_ndvi is not None else None


    obs_30d_rain = float(df_hist["PRECTOTCORR"].tail(30).sum())
    next_7d_rain = float(df_fc["fc_precip_mm"].sum())
    total_window_rain_mm = round(obs_30d_rain + next_7d_rain, 1)


    min_ds_rain = float(df_aez_mjo["rainfall"].min())
    effective_water_mm = (
        total_window_rain_mm
        if total_window_rain_mm >= min_ds_rain
        else round(max(min_ds_rain + 2.0, latest_root_h2o * 115.0), 1)
    )


    features_7 = {
        "N": latest_n_ratio,
        "P": soil_info["P_ratio"],
        "K": soil_info["K_ratio"],
        "temperature": round(latest_temp, 2),
        "humidity": round(latest_humidity, 2),
        "ph": soil_info["ph"],
        "rainfall": total_window_rain_mm,
    }


    tier1 = run_tier1_simple_decisions(
        df_hist=df_hist,
        df_fc=df_fc,
        features_7=features_7,
        soil_info=soil_info,
        terrain_info=terrain_info,
        df_crop_ext=df_aez_mjo,
        previous_crop=previous_crop,
    )
    tier2 = run_tier2_two_rotation_plans(
        df_crop_ext=df_aez_mjo,
        features_7=features_7,
        effective_water_mm=effective_water_mm,
        tier1=tier1,
        previous_crop=previous_crop,
    )
    tier3 = run_tier3_aez_mjo_connection(
        aez_code=static_aez["aez_code"],
        features_7=features_7,
        effective_water_mm=effective_water_mm,
        df_aez_mjo=df_aez_mjo,
        df_matrix_90=df_matrix_90,
        tier2=tier2,
    )


    levels = [tier1[k]["level"] for k in ("irrigation", "cover_crop", "nutrition", "insecticide")]
    if "HIGH" in levels:
        alert_level = "HIGH PRIORITY ACTION"
    elif "MID" in levels:
        alert_level = "MODERATE MONITORING"
    else:
        alert_level = "LOW / OPTIMAL"


    return {
        "alert_level": alert_level,
        "features_7": features_7,
        "effective_water_mm": effective_water_mm,
        "previous_crop": previous_crop,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "dataset_file": os.path.basename(dataset_path),
        "modis_ndvi": round(latest_ndvi, 3) if latest_ndvi is not None else "N/A",
    }




# ==============================================================================
# REUSABLE SINGLE-COORDINATE EXTRACTION PIPELINE (CALLED FOR INITIAL & EXAMPLE)
# ==============================================================================
def extract_plot_pipeline(
    lat: float,
    lon: float,
    plot_name: str,
    prev_crop: str,
    df_aez_mjo: pd.DataFrame,
    df_matrix_90: pd.DataFrame,
    dataset_path: str,
    country: str = "Target Input",
) -> dict:
    print(f"\n>>> EXTRACTING ALL SECTIONS FOR: {plot_name} ({lat}, {lon})")
    soil = get_static_soil(lat, lon)
    terrain = get_static_terrain(lat, lon)
    climate = get_static_climate_and_aez(lat, lon, soil)
    hist_df, fc_df, ndvi_df = get_dynamic_telemetry(lat, lon, soil)
    hist_df = apply_dsp_conditioning(hist_df)
    decision = run_decision_engine(
        hist_df, fc_df, ndvi_df, climate, soil, terrain, df_aez_mjo, df_matrix_90, dataset_path, prev_crop
    )
    return {
        "meta": {"lat": lat, "lon": lon, "plot_name": plot_name, "prev_crop": prev_crop, "country": country},
        "soil": soil,
        "terrain": terrain,
        "climate": climate,
        "hist_df": hist_df,
        "fc_df": fc_df,
        "ndvi_df": ndvi_df,
        "decision": decision,
    }




# ==============================================================================
# SECTION 5: COMPARATIVE EXAMINATION, GEMINI API HOOK (STUB) & result_com.csv
# ==============================================================================
def flatten_plot_record(role_tag: str, res: dict) -> dict:
    meta = res["meta"]
    soil = res["soil"]
    terr = res["terrain"]
    clim = res["climate"]
    hist_df = res["hist_df"]
    fc_df = res["fc_df"]
    dec = res["decision"]
    f7 = dec["features_7"]
    t1 = dec["tier1"]
    t2 = dec["tier2"]
    t3 = dec["tier3"]


    return {
        # SECTION 0: METADATA
        "record_role": role_tag,
        "plot_label": meta["plot_name"],
        "country": meta["country"],
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "previous_crop": meta["prev_crop"],
        "dataset_source": dec["dataset_file"],


        # SECTION 1A: STATIC SOIL
        "s1a_soil_source": soil["source"],
        "s1a_wrb_class": soil["wrb_class"],
        "s1a_clay_pct": soil["clay_pct"],
        "s1a_sand_pct": soil["sand_pct"],
        "s1a_silt_pct": soil["silt_pct"],
        "s1a_ph": soil["ph"],
        "s1a_baseline_N_g_kg": soil["baseline_nitrogen_g_kg"],
        "s1a_N_ratio": soil["N_ratio"],
        "s1a_P_ratio": soil["P_ratio"],
        "s1a_K_ratio": soil["K_ratio"],


        # SECTION 1B: STATIC TERRAIN
        "s1b_elevation_m": terr["elevation_m"],
        "s1b_slope_pct": terr["slope_pct"],
        "s1b_erosion_risk": terr["erosion_risk"],


        # SECTION 1C: STATIC CLIMATOLOGY & AEZ
        "s1c_aez_code": clim["aez_code"],
        "s1c_aez_profile": clim["aez_profile"],
        "s1c_climate_zone": clim["climate_zone"],
        "s1c_annual_temp_c": clim["annual_temp_c"],
        "s1c_annual_precip_mm": clim["annual_precip_mm"],


        # SECTION 2: DYNAMIC TELEMETRY & FORECAST
        "s2_obs_days": int(len(hist_df)),
        "s2_latest_obs_date": hist_df.index[-1].strftime("%Y-%m-%d"),
        "s2_raw_temp_c": round(float(hist_df["T2M"].iloc[-1]), 2),
        "s2_rh2m_pct": round(float(hist_df["RH2M"].iloc[-1]), 2),
        "s2_smap_surface_h2o": round(float(hist_df["GWETTOP"].iloc[-1]), 3),
        "s2_smap_root_h2o_raw": round(float(hist_df["GWETROOT"].iloc[-1]), 3),
        "s2_obs_30d_rain_mm": round(float(hist_df["PRECTOTCORR"].tail(30).sum()), 1),
        "s2_latest_et0_mm": round(float(hist_df["ET0_est_mm"].iloc[-1]), 2),
        "s2_dynamic_N_g_kg": round(float(hist_df["N2_dynamic_g_kg"].iloc[-1]), 4),
        "s2_modis_ndvi": dec["modis_ndvi"],
        "s2_fc_7d_temp_mean_c": round(float(fc_df["fc_temp_c"].mean()), 2),
        "s2_fc_7d_temp_max_c": round(float(fc_df["fc_temp_max_c"].max()), 2),
        "s2_fc_7d_rain_sum_mm": round(float(fc_df["fc_precip_mm"].sum()), 1),
        "s2_fc_7d_et0_sum_mm": round(float(fc_df["fc_et0_mm"].sum()), 1),
        "s2_fc_7d_wind_max_kmh": round(float(fc_df["fc_wind_kmh"].max()), 1),


        # SECTION 3: DSP CONDITIONING & 7-FIELD VECTOR
        "s3_dsp_temp_c": f7["temperature"],
        "s3_dsp_root_h2o": round(float(hist_df["GWETROOT_filtered"].iloc[-1]), 3),
        "s3_dsp_N_ratio": f7["N"],
        "s3_effective_water_mm": dec["effective_water_mm"],


        # SECTION 4 - TIER 1: SIMPLE DECISIONS
        "s4_t1_overall_alert": dec["alert_level"],
        "s4_t1_irrigation_level": t1["irrigation"]["level"],
        "s4_t1_irrigation_action": t1["irrigation"]["action"],
        "s4_t1_cover_crop_level": t1["cover_crop"]["level"],
        "s4_t1_cover_crop_action": t1["cover_crop"]["action"],
        "s4_t1_nutrition_level": t1["nutrition"]["level"],
        "s4_t1_nutrition_action": t1["nutrition"]["action"],
        "s4_t1_insecticide_level": t1["insecticide"]["level"],
        "s4_t1_insecticide_action": t1["insecticide"]["action"],


        # SECTION 4 - TIER 2: CROP ROTATION PLANS
        "s4_t2_starting_crop": t2["starting_crop"],
        "s4_t2_plan_a_sequence": " -> ".join(t2["plan_a"]["sequence"]),
        "s4_t2_plan_a_net_N_delta": t2["plan_a"]["net_N_delta"],
        "s4_t2_plan_b_sequence": " -> ".join(t2["plan_b"]["sequence"]),
        "s4_t2_plan_b_net_N_delta": t2["plan_b"]["net_N_delta"],


        # SECTION 4 - TIER 3: GLOBAL CONNECTION (AEZXX x MJO)
        "s4_t3_aez_mjo_key": t3["aez_mjo_key"],
        "s4_t3_prefix_pattern": t3["prefix_pattern"],
        "s4_t3_mjo_code": t3["mjo_code"],
        "s4_t3_mjo_amplitude": t3["mjo_amplitude"],
        "s4_t3_mjo_source": t3["mjo_source"],
        "s4_t3_convective_region": t3["convective_region"],
        "s4_t3_teleconnection_regime": t3["teleconnection_regime"],
        "s4_t3_global_irrigation_modifier": t3["global_irrigation_modifier"],
        "s4_t3_rain_multiplier": t3["rain_multiplier"],
        "s4_t3_temp_offset_c": t3["temp_offset_c"],
        "s4_t3_rh_offset_pct": t3["rh_offset_pct"],
        "s4_t3_benchmark_basins": t3["benchmark_basins"],
        "s4_t3_export_corridor": t3["export_corridor"],
        "s4_t3_matched_slice_rows": t3["matched_slice_rows"],
        "s4_t3_aez_total_rows": t3["aez_total_rows"],
        "s4_t3_matrix_top3_ref": t3["matrix_top3_ref"],
        "s4_t3_mjo_adjusted_top3": ", ".join([f"{c} ({p}%)" for c, p in t3["mjo_adjusted_crops"]]),
        "s4_t3_plan_a_global_score": t3["plan_a_global"]["global_score"],
        "s4_t3_plan_a_wue_pct": t3["plan_a_global"]["wue_pct"],
        "s4_t3_plan_b_global_score": t3["plan_b_global"]["global_score"],
        "s4_t3_plan_b_wue_pct": t3["plan_b_global"]["wue_pct"],
    }




def examine_and_save_comparison_csv(
    initial_res: dict,
    example_res: dict,
    match_meta: dict,
    output_csv: str = "result_com.csv",
) -> tuple[pd.DataFrame, dict]:
    """
    Examines (LON, LAT) vs (LON_ex, LAT_ex) across all sections and saves
    all extracted metrics + deltas into result_com.csv.
    """
    row_target = flatten_plot_record("TARGET_INITIAL", initial_res)
    row_example = flatten_plot_record("MATCHED_ZONE_EXAMPLE", example_res)


    # Compute normalized 7-feature similarity between Target (LON, LAT) and Example (LON_ex, LAT_ex)
    scales = np.array([60.0, 60.0, 80.0, 12.0, 35.0, 1.5, 120.0])
    v_t = np.array([
        row_target["s3_dsp_N_ratio"], row_target["s1a_P_ratio"], row_target["s1a_K_ratio"],
        row_target["s3_dsp_temp_c"], row_target["s2_rh2m_pct"], row_target["s1a_ph"],
        row_target["s3_effective_water_mm"],
    ], dtype=float)
    v_e = np.array([
        row_example["s3_dsp_N_ratio"], row_example["s1a_P_ratio"], row_example["s1a_K_ratio"],
        row_example["s3_dsp_temp_c"], row_example["s2_rh2m_pct"], row_example["s1a_ph"],
        row_example["s3_effective_water_mm"],
    ], dtype=float)
    norm_dist = float(np.sqrt(np.mean(((v_t - v_e) / scales) ** 2)))
    agronomic_similarity_pct = round(max(0.0, min(99.9, (1.0 - norm_dist) * 100.0)), 1)


    # Build Row 3: COMPARISON_DELTA (Target - Example for numeric columns, MATCH/DIFF for categoricals)
    row_delta = {}
    for col in row_target.keys():
        val_t = row_target[col]
        val_e = row_example[col]
        if col == "record_role":
            row_delta[col] = "COMPARISON_DELTA (Target - Example)"
        elif col == "plot_label":
            row_delta[col] = f"Similarity: {agronomic_similarity_pct}% | Dist: {match_meta.get('geo_dist_km', 'N/A')} km"
        elif isinstance(val_t, (int, float)) and isinstance(val_e, (int, float)) and not isinstance(val_t, bool):
            row_delta[col] = round(float(val_t) - float(val_e), 3)
        else:
            row_delta[col] = "MATCH" if str(val_t) == str(val_e) else f"DIFF ({val_t} vs {val_e})"


    df_com = pd.DataFrame([row_target, row_example, row_delta])
    df_com.insert(1, "agronomic_similarity_pct", [100.0, agronomic_similarity_pct, agronomic_similarity_pct])
    df_com.insert(2, "example_match_mode", ["Initial Input", match_meta["selection_mode"], "Comparison Summary"])


    out_path = os.path.abspath(output_csv)
    df_com.to_csv(out_path, index=False)
    print(f"\n[SAVED] All data across Sections 0-4 saved to: {out_path} ({len(df_com)} rows x {len(df_com.columns)} cols)")


    examination_summary = {
        "target_coords": (row_target["latitude"], row_target["longitude"]),
        "example_coords": (row_example["latitude"], row_example["longitude"]),
        "example_name": f"{match_meta['city_basin']}, {match_meta['country']}",
        "aez_match": f"{row_target['s1c_aez_code']} vs {row_example['s1c_aez_code']}",
        "aez_mjo_match": f"{row_target['s4_t3_aez_mjo_key']} vs {row_example['s4_t3_aez_mjo_key']}",
        "agronomic_similarity_pct": agronomic_similarity_pct,
        "delta_temp_c": row_delta["s3_dsp_temp_c"],
        "delta_rh_pct": row_delta["s2_rh2m_pct"],
        "delta_root_h2o": row_delta["s3_dsp_root_h2o"],
        "delta_N_ratio": row_delta["s3_dsp_N_ratio"],
        "delta_ph": row_delta["s1a_ph"],
        "irrigation_parity": row_delta["s4_t1_irrigation_level"],
        "nutrition_parity": row_delta["s4_t1_nutrition_level"],
        "plan_a_parity": row_delta["s4_t2_plan_a_sequence"],
    }
    return df_com, examination_summary




def prepare_gemini_comparison_payload(initial_res: dict, example_res: dict, exam_summary: dict) -> dict:
    """
    Placeholder hook structuring the examined (LON, LAT) and (LON_ex, LAT_ex) outputs
    for future Gemini API synthesis.
    [GEMINI API CALL INTENTIONALLY EXCLUDED BY NOW]
    """
    return {
        "status": "READY_FOR_GEMINI_API_LATER",
        "examination_summary": exam_summary,
        "target_initial_outputs": flatten_plot_record("TARGET_INITIAL", initial_res),
        "matched_example_outputs": flatten_plot_record("MATCHED_ZONE_EXAMPLE", example_res),
    }




# ==============================================================================
# SECTION 6: COMPARATIVE 4-PANEL DASHBOARD & PNG GENERATION
# ==============================================================================
def plot_comparative_dashboard(
    initial_res: dict,
    example_res: dict,
    exam_summary: dict,
    no_show: bool = False,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(12.4, 14.8), gridspec_kw={"height_ratios": [0.95, 0.95, 0.88, 1.48]})


    h_t, fc_t = initial_res["hist_df"], initial_res["fc_df"]
    h_e, fc_e = example_res["hist_df"], example_res["fc_df"]
    t_name = initial_res["meta"]["plot_name"]
    e_name = example_res["meta"]["plot_name"]
    t_dec = initial_res["decision"]
    e_dec = example_res["decision"]


    fig.suptitle(
        f"SiriusTa Comparative Examination: {t_name} vs {e_name} (Similarity: {exam_summary['agronomic_similarity_pct']}%)\n"
        f"Target [{t_dec['tier3']['aez_mjo_key']}] ({initial_res['meta']['lat']}, {initial_res['meta']['lon']})  |  "
        f"Matched Twin [{e_dec['tier3']['aez_mjo_key']}] ({example_res['meta']['lat']}, {example_res['meta']['lon']})",
        fontsize=9.8,
        fontweight="bold",
        y=0.988,
    )


    # Panel 1: Temperature Comparison
    ax1 = axes[0]
    ax1.plot(h_t.index, h_t["T2M_filtered"], color="#c0392b", linewidth=2.3, label=f"Target DSP Temp: {t_name} (C)")
    ax1.plot(h_e.index, h_e["T2M_filtered"], color="#2980b9", linestyle="--", linewidth=2.0, label=f"Matched Twin DSP Temp: {e_name} (C)")
    ax1.plot(fc_t.index, fc_t["fc_temp_c"], color="#e67e22", marker="o", linewidth=1.8, label="Target 7d Forecast (C)")
    ax1.set_ylabel("Temp (C)", fontweight="bold")
    ax1.set_title("1. Dynamic Temperature Comparison (Target vs Matched Zone Example)", loc="left", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)


    # Panel 2: Soil H2O & Rainfall Comparison
    ax2 = axes[1]
    l1 = ax2.plot(h_t.index, h_t["GWETROOT_filtered"], color="#1f618d", linewidth=2.4, label=f"Target Root H2O: {t_name}")
    l2 = ax2.plot(h_e.index, h_e["GWETROOT_filtered"], color="#16a085", linestyle="--", linewidth=2.0, label=f"Twin Root H2O: {e_name}")
    l3 = ax2.axhline(0.42, color="#e74c3c", linestyle=":", linewidth=1.4, label="HIGH Irrigation Threshold (0.42)")
    ax2.set_ylabel("Soil H2O Index (0-1)", fontweight="bold", color="#1f618d")
    ax2.set_ylim(0, 1.0)


    ax2_r = ax2.twinx()
    b1 = ax2_r.bar(h_t.index, h_t["PRECTOTCORR"], color="#2ecc71", alpha=0.4, width=0.6, label="Target Rain (mm)")
    b2 = ax2_r.bar(fc_t.index, fc_t["fc_precip_mm"], color="#1abc9c", alpha=0.75, width=0.6, label="Target 7d Fcst Rain (mm)")
    ax2_r.set_ylabel("Rain (mm/day)", fontweight="bold", color="#16a085")
    ax2_r.set_ylim(0, max(15, max(h_t["PRECTOTCORR"].max(), h_e["PRECTOTCORR"].max()) * 1.35))
    ax2.set_title("2. Root Zone Moisture (SMAP) & Precipitation Comparison", loc="left", fontsize=10, fontweight="bold")
    handles = l1 + l2 + [l3, b1, b2]
    ax2.legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=True, fontsize=8.5)


    # Panel 3: Dynamic Soil Nitrogen Comparison
    ax3 = axes[2]
    ax3.plot(h_t.index, h_t["N_ratio_filtered"], color="#5b2c6f", linewidth=2.4, label=f"Target Soil N-Ratio: {t_name}")
    ax3.plot(h_e.index, h_e["N_ratio_filtered"], color="#af7ac5", linestyle="--", linewidth=2.0, label=f"Twin Soil N-Ratio: {e_name}")
    ax3.set_ylabel("Soil N Ratio (0-140)", fontweight="bold", color="#5b2c6f")
    ax3.set_title("3. Dynamic Soil Nitrogen (N-Ratio) Trajectory Comparison", loc="left", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, fontsize=8.5)


    # Panel 4: Side-by-Side Examination & result_com.csv Summary
    ax4 = axes[3]
    ax4.axis("off")
    t1_t, t2_t, t3_t = t_dec["tier1"], t_dec["tier2"], t_dec["tier3"]
    t1_e, t2_e, t3_e = e_dec["tier1"], e_dec["tier2"], e_dec["tier3"]
    summary_text = (
        f"DUAL-COORDINATE EXAMINATION SUMMARY (Saved to result_com.csv)\n"
        f"  * Target (LAT, LON)       : ({initial_res['meta']['lat']}, {initial_res['meta']['lon']}) — {t_name} [{t3_t['aez_mjo_key']}]\n"
        f"  * Matched Twin (LATex,LONex): ({example_res['meta']['lat']}, {example_res['meta']['lon']}) — {e_name} [{t3_e['aez_mjo_key']}]\n"
        f"  * Agronomic Similarity    : {exam_summary['agronomic_similarity_pct']}% | Deltas (T-Ex): dTemp={exam_summary['delta_temp_c']:+.2f}C, dRH={exam_summary['delta_rh_pct']:+.1f}%, dH2O={exam_summary['delta_root_h2o']:+.3f}\n\n"
        f"TIER 1 & TIER 2 COMPARISON\n"
        f"  * Target Tier 1           : Irrig={t1_t['irrigation']['level']} | Cover={t1_t['cover_crop']['level']} | Nutr={t1_t['nutrition']['level']} | Pest={t1_t['insecticide']['level']}\n"
        f"  * Twin   Tier 1           : Irrig={t1_e['irrigation']['level']} | Cover={t1_e['cover_crop']['level']} | Nutr={t1_e['nutrition']['level']} | Pest={t1_e['insecticide']['level']}\n"
        f"  * Target Plan A / Plan B  : {' -> '.join(t2_t['plan_a']['sequence'])}  ||  {' -> '.join(t2_t['plan_b']['sequence'])}\n"
        f"  * Twin   Plan A / Plan B  : {' -> '.join(t2_e['plan_a']['sequence'])}  ||  {' -> '.join(t2_e['plan_b']['sequence'])}\n\n"
        f"TIER 3 GLOBAL TELECONNECTION & EXPORT CORRIDOR\n"
        f"  * Active MJO Regime       : {t3_t['mjo_code']} ({t3_t['convective_region']}) -> {t3_t['global_irrigation_modifier']}\n"
        f"  * Export Corridor         : {t3_t['export_corridor']} | Gemini Hook: READY (Excluded by now)"
    )
    ax4.text(
        0.01,
        0.5,
        summary_text,
        fontsize=8.2,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.75", facecolor="#f4f6f7", edgecolor="#2c3e50", linewidth=1.5),
    )


    plt.tight_layout(rect=[0, 0, 1, 0.935])
    output_file = os.path.abspath("siriusta_poc_output.png")
    plt.savefig(output_file, dpi=300)
    print(f"[PLOT SAVED] {output_file}")


    if not no_show:
        plt.show()




# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    # 0. Get initial (LAT, LON) and load AEZXX_MJO dataset, 90-cell matrix, and 12_ex.csv
    lat, lon, plot_name, prev_crop, ex_row, no_show = get_user_inputs()
    df_aez_mjo, df_matrix_90, df_examples_12, dataset_path = load_local_datasets()


    # 1. Extract all section outputs for initial (LAT, LON)
    initial_outputs = extract_plot_pipeline(
        lat=lat,
        lon=lon,
        plot_name=plot_name,
        prev_crop=prev_crop,
        df_aez_mjo=df_aez_mjo,
        df_matrix_90=df_matrix_90,
        dataset_path=dataset_path,
        country="Target Input",
    )


    # 2. Match AEZ zone in 12_ex.csv and pass (LAT_ex, LON_ex) to the SAME extraction pipeline
    matched_ex = select_matched_zone_example(
        target_lat=lat,
        target_lon=lon,
        target_aez_code=initial_outputs["climate"]["aez_code"],
        df_examples=df_examples_12,
        preferred_row_num=ex_row,
    )
    print("\n" + "-" * 82)
    print(
        f"[MATCHED ZONE EXAMPLE] Row #{matched_ex['ex_id']}: {matched_ex['city_basin']}, {matched_ex['country']} "
        f"({matched_ex['lat_ex']}, {matched_ex['lon_ex']}) | {matched_ex['selection_mode']}"
    )
    print("-" * 82)


    example_outputs = extract_plot_pipeline(
        lat=matched_ex["lat_ex"],
        lon=matched_ex["lon_ex"],
        plot_name=f"{matched_ex['city_basin']} ({matched_ex['country']})",
        prev_crop=prev_crop,
        df_aez_mjo=df_aez_mjo,
        df_matrix_90=df_matrix_90,
        dataset_path=dataset_path,
        country=matched_ex["country"],
    )


    # 3. Examine (LON, LAT) vs (LON_ex, LAT_ex), save all sections to result_com.csv, and stage Gemini hook
    df_result_com, exam_summary = examine_and_save_comparison_csv(
        initial_res=initial_outputs,
        example_res=example_outputs,
        match_meta=matched_ex,
        output_csv="result_com.csv",
    )
    gemini_ready_payload = prepare_gemini_comparison_payload(initial_outputs, example_outputs, exam_summary)


    # 4. Print Comparative Console Examination Report
    print("\n" + "=" * 82)
    print("COMPARATIVE EXAMINATION REPORT: (LON, LAT) vs (LON_ex, LAT_ex)")
    print("=" * 82)
    print(f"  Target Initial   : {plot_name} -> ({lat}, {lon}) [{exam_summary['aez_mjo_match'].split(' vs ')[0]}]")
    print(f"  Matched Example  : {exam_summary['example_name']} -> ({matched_ex['lat_ex']}, {matched_ex['lon_ex']}) [{exam_summary['aez_mjo_match'].split(' vs ')[1]}]")
    print(f"  7-Field Similarity: {exam_summary['agronomic_similarity_pct']}%")
    print(f"  Telemetry Deltas : dTemp={exam_summary['delta_temp_c']:+.2f} C | dRH={exam_summary['delta_rh_pct']:+.1f}% | dRootH2O={exam_summary['delta_root_h2o']:+.3f} | dN={exam_summary['delta_N_ratio']:+.1f} | dpH={exam_summary['delta_ph']:+.2f}")
    print(f"  Decision Parity  : Irrigation={exam_summary['irrigation_parity']} | Nutrition={exam_summary['nutrition_parity']} | PlanA={exam_summary['plan_a_parity']}")
    print(f"  CSV Export       : result_com.csv ({len(df_result_com)} rows x {len(df_result_com.columns)} columns)")
    print(f"  Gemini API Hook  : {gemini_ready_payload['status']} (Not executed yet)")
    print("=" * 82)


    plot_comparative_dashboard(initial_outputs, example_outputs, exam_summary, no_show=no_show)