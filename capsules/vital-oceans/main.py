# main.py

from typing import Union, Literal, List

import os
import io
import json
import asyncio
import tempfile
import zipfile

from pathlib import Path
from math import sqrt, pi
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import geopandas as gpd
from pandasql import sqldf

import shapely
import xarray as xr
from shapely import MultiPolygon, MultiPoint, Polygon, Point

import lakefs
from lakefs.client import Client

from fastmcp import FastMCP, Context
from starlette.requests import Request
from starlette.responses import PlainTextResponse

dependencies = [
    "httpx",
    "geopandas",
    "shapely",
    "lakefs",
    "xarray[io]",
    "pyproj",
    "numpy",
    "pandas",
    "pandasql",
    "mo_sql_parsing",
]

mcp = FastMCP(name="vital-oceans", dependencies=dependencies)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

# --- Global Data Storage --- #
CACHE = {}
DATA_DIR = Path("./mcp_data")

# --- lakeFS Client --- #
load_dotenv()
LAKEFS_HOST = os.environ["LAKEFS_HOST"]
LAKEFS_USERNAME = os.environ["LAKEFS_USERNAME"]
LAKEFS_PASSWORD = os.environ["LAKEFS_PASSWORD"]
LAKEFS_REPO_ID = os.getenv("LAKEFS_REPO_ID", "vital-oceans")

lakefs_client = Client(
    username=LAKEFS_USERNAME,
    password=LAKEFS_PASSWORD,
    host=LAKEFS_HOST,
)

repo = lakefs.Repository(repository_id=LAKEFS_REPO_ID, client=lakefs_client)
ref = repo.ref("main")

def get_available_paths(data_type: str) -> list[str]:
    """
    Get all available paths for a given data type (shapefile or netcdf).
    Returns a list of available dataset paths relative to the data type folder.
    """
    try:
        available_paths = []
        base_path = f"shapefiles/"
        
        # Get all objects in the data type directory
        all_objects = list(ref.objects(prefix=base_path))
        
        if data_type == "shapefile":
            # For shapefiles, we need to find .shp files specifically
            shapefile_paths = set()
            
            for obj in all_objects:
                path = obj.path
                
                # Check if it's a .shp file (main shapefile component)
                if path.endswith('.shp'):
                    # Get the relative path from the base_path
                    relative_path = path[len(base_path):]
                    shapefile_paths.add(relative_path)
            
            available_paths = sorted(list(shapefile_paths))
            
        elif data_type == "NetCDFs":
            # For NetCDF files, look for .nc files
            netcdf_paths = set()
            
            for obj in all_objects:
                path = obj.path
                
                # Check if it's a .nc file
                if path.endswith('.nc'):
                    # Get the relative path from the base_path
                    relative_path = path[len(base_path):]
                    netcdf_paths.add(relative_path)
            
            available_paths = sorted(list(netcdf_paths))
        
        return available_paths
        
    except Exception as e:
        print(f"Error getting available paths for {data_type}: {e}")
        return []

# --- Download from LafeFS --- #

def download_and_cache_shapefile(sub_path: str) -> dict:
    """
    Downloads shapefile components from LakeFS and saves them to persistent storage.
    Returns paths to the main .shp file and its schema.
    """
    lakefs_path = Path("shapefiles") / sub_path
    local_path = DATA_DIR / "shapefiles" / sub_path
    local_dir = local_path.parent

    # Create local dir
    if ref.object(str(lakefs_path)).exists():
        local_dir.mkdir(parents=True, exist_ok=True)

    # Get shapefile components
    shapefile_extensions = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".csv", ".json"}
    prefix = str(lakefs_path.parent)

    shapefile_objects = []
    for obj in ref.objects(prefix=prefix):
        fname = os.path.basename(obj.path)
        if fname.startswith(lakefs_path.stem) and Path(fname).suffix.lower() in shapefile_extensions:
            shapefile_objects.append(obj)

    if not shapefile_objects:
        raise FileNotFoundError(f"No shapefile components found at: {prefix}")

    # Download each file if it doesn't already exist or is empty
    for obj in shapefile_objects:
        remote_path = obj.path
        basename = os.path.basename(remote_path)
        file_path = local_dir / basename

        if file_path.exists() and file_path.stat().st_size > 0:
            continue

        with ref.object(remote_path).reader(mode="rb") as r:
            with open(file_path, "wb") as f:
                f.write(r.read())

    # TODO: This error should raise if one or more of the non-optional shape components is missing
    if not local_path.exists():
        raise FileNotFoundError(f"Main .shp file not downloaded: {local_path}")

    local_paths = {
        "shapefile": local_path,
        "schema": local_path.parent / (str(local_path.stem) + ".json")
    }
    return local_paths

def download_and_cache_netcdf(sub_path: str) -> Path:
    """
    Downloads a NetCDF file from LakeFS and saves it to persistent storage.
    Returns the local path to the NetCDF file.
    """
    local_path = DATA_DIR / "NetCDFs" / sub_path
    lakefs_path = Path("NetCDFs") / sub_path

    try:
        if ref.object(str(lakefs_path)).exists():
            # Create local dir, if not already exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file already exists and is not empty
            if local_path.exists() and local_path.stat().st_size > 0:
                return local_path

            # Download it
            with ref.object(str(lakefs_path)).reader(mode="rb") as r:
                with open(local_path, "wb") as f:
                    f.write(r.read())
        else:
            raise FileNotFoundError(f"No NetCDF file found at: {sub_path}")
    except Exception as e:
        raise e

    return local_path

def load_shapefile_to_gdf(sub_path: str) -> dict:
    """
    Loads a shapefile from cache into a GeoDataFrame.
    """
    if sub_path not in CACHE:
        local_paths = download_and_cache_shapefile(sub_path)

        # Read and procees data files
        schema = {}
        try:
            with open(local_paths["schema"]) as f:
                schema = json.load(f)
        except FileNotFoundError:
            pass

        gdf = gpd.read_file(local_paths["shapefile"]).to_crs("EPSG:4326")

        # Add to cache 
        CACHE[sub_path] = gdf, schema
        print(f"Loaded and cached shapefile data at: {sub_path}")
    
    return CACHE[sub_path]

def load_netcdf_data(sub_path: str) -> dict:
    """
    Loads NetCDF data from cache and processes it.
    """
    
    if sub_path not in CACHE:
        local_path = download_and_cache_netcdf(sub_path)
        
        # Procees data file
        nc = xr.open_dataset(local_path, engine="h5netcdf")
        
        lats = nc['lat'].values
        lons = nc['lon'].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 0], points[:, 1]))
        
        key = list(nc.data_vars)[0]
        
        processed_data = {
            "description": nc[key].attrs.get("long_name"),
            "start_time": nc.attrs.get("start_time"),
            "end_time": nc.attrs.get("end_time"),
            "values": nc[key].values,
            "units": nc[key].attrs.get("units"),
            "key": key,
            "gdf": gdf
        }
        
        nc.close()  # Close the dataset to free memory

        # Add to cache
        CACHE[sub_path] = processed_data
        print(f"Loaded and cached NetCDF data at: {sub_path}")
    
    return CACHE[sub_path]

async def initialize_data():
    """
    Initialize and cache commonly used datasets at startup.
    """
    print("Initializing data cache...")
    
    # Get available datasets
    available_shapefiles = get_available_paths("shapefile")
    available_netcdfs = get_available_paths("netcdf")
    
    print(f"Available shapefiles: {available_shapefiles}")
    print(f"Available netcdfs: {available_netcdfs}")
    
    # Pre-load common files to avoid long startup times
    common_shapefiles = [
        'species/species_richness/species_richness.shp',
        'species/red_list/IUCN/IUCN.shp',
        'ecosystems/coldwater_coralreefs/coldwater_coralreefs.shp',
        'ecosystems/warmwater_coralreefs/warmwater_coralreefs.shp',
        'ecosystems/mangroves/mangroves.shp',
        'ecosystems/disturbed_mangroves/disturbed_mangroves.shp',
        'ecosystems/wetlands/wetlands.shp',
        'ecosystems/kelp/kelp.shp',
        'ecosystems/seagrass/seagrass.shp',
        'socioeconomic/fishing_exploitation_areas/mollusks/mollusks.shp',
        'socioeconomic/fishing_exploitation_areas/crustaceans/crustaceans.shp',
        'socioeconomic/fishing_exploitation_areas/squid/squid.shp',
        'socioeconomic/fishing_exploitation_areas/scale/scale.shp',
        'socioeconomic/fishing_exploitation_areas/echinoderms/echinoderms.shp',
        'socioeconomic/fishing_exploitation_areas/sardines/sardines.shp',
        'socioeconomic/fishing_exploitation_areas/cartilaginous/cartilaginous.shp',
        'socioeconomic/fishing_exploitation_areas/shrimp/shrimp.shp',
    ]
    for shapefile in common_shapefiles:
        if shapefile in available_shapefiles:
            try:
                load_shapefile_to_gdf(shapefile)
            except Exception as e:
                print(f"Warning: Could not pre-load shapefile {shapefile}: {e}")
    
    common_netcdfs = []
    for netcdf in common_netcdfs:
        if netcdf in available_netcdfs:
            try:
                load_netcdf_data(netcdf)
            except Exception as e:
                print(f"Warning: Could not pre-load NetCDF {netcdf}: {e}")
    
    print("Data cache initialization complete.")

# --- Resource handlers --- #

@mcp.resource("lakefs://shapefiles/{sub_path}")
def get_shapefile_binary(sub_path: str, context: Context):
    """
    Returns binary content of a zipped shapefile from persistent storage.
    """
    local_path = DATA_DIR / "shapefile" / sub_path
    
    if not local_path.exists():
        # Fallback: download if not cached
        download_and_cache_shapefile(sub_path)
    
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in local_path.iterdir():
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

@mcp.resource("lakefs://NetCDFs/{sub_path}")
def get_netcdf_binary(sub_path: str, context: Context):
    """
    Returns binary content of a NetCDF file from persistent storage.
    """
    local_path = DATA_DIR / "NetCDFs" / sub_path
    
    if not local_path.exists():
        # Fallback: download if not cached
        download_and_cache_netcdf(sub_path)
    
    with open(local_path, "rb") as f:
        return f.read()

# --- Tools --- #

from utils import parse_coordinates, get_shp_paths, calculate_geodesic_area, generate_report_from_gdf

@mcp.tool()
async def calculate_polygon_area(
    polygon_coordinates: list[dict],
    context: Context) -> float:
    """Calculate the geodesic area of a polygon.

    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinates pairs representing the vertices of the polygon.
        Format: [{"lat": float, "lng": float}, ...]

    Returns
    -------
    float
        Area in km²."""

    # Define polygon
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    return calculate_geodesic_area(polygon)


@mcp.tool()
async def get_red_list_in_polygon(
    data_source: Literal["IUCN", "OBIS"],
    polygon_coordinates: list[dict],
    context: Context,
) -> str:
    """Query the IUCN or OBIS red list for endangered marine species within a polygon.

    Parameters
    ----------
    data_source : str
        The source of species data to query.
        Supported options: "IUCN", "OBIS".
    polygon_coordinates : list of dict
        List of coordinates pairs representing the vertices of the polygon.
        Format: [{"lat": float, "lng": float}, ...]
    
    Returns
    -------
    str
        Markdown-formatted report of observed species and their threat categories."""

    # Set shapefiles paths
    shapefile_map = {
        "IUCN": "species/red_list/IUCN/IUCN.shp",
        "OBIS": "species/red_list/OBIS/OBIS.shp"
    }

    if data_source not in shapefile_map:
        raise ValueError(f"Unsupported data source: '{data_source}'. Supported options: {list(shapefile_map.keys())}")
    shp_paths = [shapefile_map[data_source]]

    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)

    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    # Generate reports
    reports = ""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,

        )

    return reports


@mcp.tool()
async def report_key_area_overlap(
    polygon_coordinates: list[dict],
    context: Context,
    buffer_km: float = 0.0,
) -> str:
    """Generate a report on key marine areas that overlap with a given polygon.
    Key areas include: Biological Primary Productivity Areas, Key Biodiversity Areas, Protected Conservation Areas and Wetlands Ramsar.
    
    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinates pairs representing the vertices of the polygon.
        Format: [{"lat": float, "lng": float}, ...]
    buffer_km : float (optional)
        Optional buffer (in kilometers) to expand the polygon before analysis.
    
    Returns
    -------
    str
        Markdown-formatted report summarizing key area overlaps."""

    # Get shapefile paths
    shp_paths = get_shp_paths(dir="key_areas", ref=ref)
    
    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)

    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    # Calculate polygon area
    area = calculate_geodesic_area(polygon)

    # Generate reports
    reports = f"""# Key Area Overlap Analysis
The provided polygon covers an area of {area:,.2f} km².
Below is an analysis of the key areas that overlap with this polygon.
"""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,
            buffer_km=buffer_km
        )

    return reports


@mcp.tool()
async def report_ecosystem_overlap(
    polygon_coordinates: list[dict],
    context: Context,
) -> str:
    """Generate a report on ecosystems that overlap with a given polygon.
    Ecosystems include: cold- and warm-water coral reefs, kelp, seagrass, wetlands, mangroves, and disturbed mangroves.
    
    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinates pairs representing the vertices of the polygon.
        Format: [{"lat": float, "lng": float}, ...]
    
    Returns
    -------
    str
        Markdown-formatted report summarizing ecosystem overlaps."""

    # Get shapefile paths
    shp_paths = get_shp_paths(dir="ecosystems", ref=ref)
    
    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)

    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    # Calculate polygon area
    area = calculate_geodesic_area(polygon)

    # Generate reports
    reports = f"""# Ecosystem Overlap Analysis
The provided polygon covers an area of {area:,.2f} km².
Below is an analysis of the ecosystems that overlap with this polygon.
"""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,
        )

    return reports


@mcp.tool()
async def report_exploitation_area_overlap(
    polygon_coordinates: list[dict],
    context: Context,
) -> str:
    """Generate a report on fishing exploitation areas that overlap with a given polygon.
    Exploitation types include: cartilaginous, crustaceans, echinoderms, mollusks, sardines, scale, shrimp and squid.
    
    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinates pairs representing the vertices of the polygon.
        Format: [{"lat": float, "lng": float}, ...]
    
    Returns
    -------
    str
        Markdown-formatted report summarizing fishing exploitation area overlaps."""

    # Get shapefile paths
    shp_paths = get_shp_paths(dir="socioeconomic/fishing_exploitation_areas", ref=ref)
    
    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)

    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    # Calculate polygon area
    area = calculate_geodesic_area(polygon)

    # Generate reports
    reports = f"""# Fishing Exploitation Area Overlap Analysis
The provided polygon covers an area of {area:,.2f} km².
Below is an analysis of the fishing exploitation areas that overlap with this polygon.
"""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,
        )

    return reports


@mcp.tool()
async def find_nearby_coastal_communities(
    polygon_coordinates: list[dict],
    context: Context,
    buffer_km: float = 0.0,
) -> str:
    """Generate a report of coastal communities located within or near a given polygon.
    
    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinate pairs representing the polygon vertices.
        Format: [{"lat": float, "lng": float}, ...]
    buffer_km : float (optional)
        Optional buffer (in kilometers) to expand the polygon for proximity search.
    
    Returns
    -------
    str
        Markdown-formatted report of nearby coastal communities and their population."""

    # Set shapefile paths
    shp_paths = ["socioeconomic/coastal_communities/coastal_communities.shp"]
    
    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)

    # Generate reports
    reports = ""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,
            buffer_km=buffer_km
        )

    return reports


@mcp.tool()
async def get_human_activity_in_polygon(
    polygon_coordinates: list[dict],
    context: Context,
) -> str:
    """Generate a report summarizing human activities within a given polygon.
    Human activities include: diving sites, fishing refuges, and sport fishing areas.
    
    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinate pairs representing the polygon vertices.
        Format: [{"lat": float, "lng": float}, ...]
    
    Returns
    -------
    str
        Markdown-formatted report of human activities found within the specified area."""

    # Get shapefile paths
    shp_paths = get_shp_paths(dir="socioeconomic/human_activity", ref=ref)
    
    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)

    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    # Calculate polygon area
    area = calculate_geodesic_area(polygon)

    # Generate reports
    reports = f"""# Human Activity report
The provided polygon covers an area of {area:,.2f} km².
This report summarizes the various human activities occurring within this area.
"""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,
        )

    return reports


@mcp.tool()
async def calculate_social_lag_in_polygon(
    polygon_coordinates: list[dict],
    context: Context,
) -> str:
    """Calculate social lag within a given polygon.
    The social lag index measures levels of deprivation across key dimensions of well-being.
    
    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinate pairs representing the polygon vertices.
        Format: [{"lat": float, "lng": float}, ...]
    
    Returns
    -------
    str
        Average social lag index found within the specified area."""

    # Set shapefile paths
    shp_paths = ["socioeconomic/social_lag/social_lag.shp"]
        
    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)
    
    # Generate reports
    reports = ""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,
        )

    return reports


@mcp.tool()
async def get_fauna_in_polygon(
    polygon_coordinates: list[dict],
    context: Context,
) -> str:
    """
    Generate a report on species richness within a given polygon.
    
    Parameters
    ----------
    polygon_coordinates : list of dict
        List of coordinate pairs representing the polygon vertices.
        Format: [{"lat": float, "lng": float}, ...]
    
    Returns
    -------
    str
        Markdown-formatted report summarizing species richness (fauna) in the specified area.
    """

    # Set shapefile paths
    shp_paths = ["species/species_richness/species_richness.shp"]

    # Build base polygon from coordinates
    _ = parse_coordinates(polygon_coordinates)
    polygon = Polygon(_)
    
    # Generate reports
    reports = ""
    for shp_path in shp_paths:
        # Load gdf and schema
        gdf, schema = load_shapefile_to_gdf(sub_path=shp_path)

        if not schema:
            if len(shp_paths)==1:
                raise ValueError(f"Metadata is missing for {shp_path}")
            else:
                continue

        reports += generate_report_from_gdf(
            shp_path=shp_path,
            schema=schema,
            gdf=gdf,
            polygon=polygon,
        )

    return reports


# --- Run MCP --- #

if __name__ == "__main__":
    # Initialize data cache before starting the server
    asyncio.run(initialize_data())
    
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        log_level="info",
    )