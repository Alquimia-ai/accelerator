from dotenv import load_dotenv
from pathlib import Path
import tempfile
import zipfile
import asyncio
import os
import io

import numpy as np
import pandas as pd
import pandasql as ps
import geopandas as gpd

from shapely.geometry import Polygon
from pyproj import Geod
import xarray as xr

from lakefs.client import Client
from lakefs import Repository

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
    host=LAKEFS_HOST,
    username=LAKEFS_USERNAME,
    password=LAKEFS_PASSWORD,
)

repo = Repository(repository_id=LAKEFS_REPO_ID, client=lakefs_client)
ref = repo.ref("main")

def get_available_paths(data_type: str) -> list[str]:
    """
    Get all available paths for a given data type (csv, shapefile, netcdf).
    Returns a list of available dataset paths relative to the data type folder.
    """
    try:
        available_paths = []
        base_path = f"datasets/{data_type}/"
        
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
            
        elif data_type == "netcdf":
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
            
        elif data_type == "csv":
            # For CSV files, look for .csv files
            csv_paths = set()
            
            for obj in all_objects:
                path = obj.path
                
                # Check if it's a .csv file
                if path.endswith('.csv'):
                    # Get the relative path from the base_path
                    relative_path = path[len(base_path):]
                    csv_paths.add(relative_path)
            
            available_paths = sorted(list(csv_paths))
        
        return available_paths
        
    except Exception as e:
        print(f"Error getting available paths for {data_type}: {e}")
        return []

# --- Download from LafeFS --- #

def download_and_cache_shapefile(sub_path: str) -> Path:
    """
    Downloads shapefile components from LakeFS and saves them to persistent storage.
    Returns the path to the main .shp file.
    """
    lakefs_path = Path("datasets/shapefile") / sub_path
    local_path = DATA_DIR / "shapefile" / sub_path
    local_dir = local_path.parent

    # Create local dir
    if ref.object(str(lakefs_path)).exists():
        local_dir.mkdir(parents=True, exist_ok=True)

    # Get shapefile components
    shapefile_extensions = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".csv"}
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

    if not local_path.exists():
        raise FileNotFoundError(f"Main .shp file not downloaded: {local_path}")

    return local_path

def download_and_cache_netcdf(sub_path: str) -> Path:
    """
    Downloads a NetCDF file from LakeFS and saves it to persistent storage.
    Returns the local path to the NetCDF file.
    """
    local_path = DATA_DIR / "netcdf" / sub_path
    lakefs_path = Path("datasets/netcdf") / sub_path

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

def download_and_cache_csv(sub_path: str) -> Path:
    """
    Downloads a CSV file from LakeFS and saves it to persistent storage.
    Returns the local path to the CSV file.
    """
    local_path = DATA_DIR / "csv" / sub_path
    lakefs_path = Path("datasets/csv") / sub_path

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
            raise FileNotFoundError(f"No CSV file found at: {sub_path}")
    except Exception as e:
        raise e

    return local_path

def load_shapefile_to_gdf(sub_path: str) -> gpd.GeoDataFrame:
    """
    Loads a shapefile from cache into a GeoDataFrame.
    """
    if sub_path not in CACHE:
        local_path = download_and_cache_shapefile(sub_path)

        # Procees data file
        gdf = gpd.read_file(local_path).to_crs("EPSG:4326")
        gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]

        # Add to cache 
        CACHE[sub_path] = gdf
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

def load_csv_to_df(sub_path: str) -> pd.DataFrame:
    """
    Loads a CSV from cache into a DataFrame.
    """
    if sub_path not in CACHE:
        local_path = download_and_cache_csv(sub_path)
        
        # Process data file
        df = pd.read_csv(
            filepath_or_buffer=local_path,
            sep=","
        )

        # Add to cache
        CACHE[sub_path] = df
        print(f"Loaded and cached CSV data at: {sub_path}")
    
    return CACHE[sub_path]

async def initialize_data():
    """
    Initialize and cache commonly used datasets at startup.
    """
    print("Initializing data cache...")
    
    # Get available datasets
    available_shapefiles = get_available_paths("shapefile")
    available_netcdfs = get_available_paths("netcdf")
    available_csvs = get_available_paths("csv")
    
    print(f"Available shapefiles: {available_shapefiles}")
    print(f"Available netcdfs: {available_netcdfs}")
    print(f"Available csvs: {available_csvs}")
    
    # Pre-load common files to avoid long startup times

    common_shapefiles = [
        "socioeconomico/pesca.shp",
        "socioeconomico/turismo.shp",
        "meteorologia/ciclones",
        "habitats/cenotes",
        "habitats/corales",
        "habitats/humedales",
        "habitats/kelp",
        "habitats/manglares",
        ]
    for shapefile in common_shapefiles:
        if shapefile in available_shapefiles:
            try:
                load_shapefile_to_gdf(shapefile)
            except Exception as e:
                print(f"Warning: Could not pre-load shapefile {shapefile}: {e}")
    
    common_netcdfs = [
        "temperature.nc",
        "chlorophylla.nc",
        "humidity.nc",
        ]
    for netcdf in common_netcdfs:
        if netcdf in available_netcdfs:
            try:
                load_netcdf_data(netcdf)
            except Exception as e:
                print(f"Warning: Could not pre-load NetCDF {netcdf}: {e}")

    common_csvs = [
        "tablas_biodiversidad/anelidos_bahia_lapaz.csv",
        "tablas_biodiversidad/anfibia_bahia_lapaz.csv",
        "tablas_biodiversidad/autotrofos_bahia_lapaz.csv",
        "tablas_biodiversidad/aves_bahia_lapaz.csv",
        "tablas_biodiversidad/cnidaria_bahia_lapaz.csv",
    ]
    for csv in common_csvs:
        if csv in available_csvs:
            try:
                load_netcdf_data(csv)
            except Exception as e:
                print(f"Warning: Could not pre-load CSV {csv}: {e}")
    
    print("Data cache initialization complete.")

# --- Resource handlers --- #

@mcp.resource("lakefs://shapefile/{sub_path}")
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

@mcp.resource("lakefs://netcdf/{sub_path}")
def get_netcdf_binary(sub_path: str, context: Context):
    """
    Returns binary content of a NetCDF file from persistent storage.
    """
    local_path = DATA_DIR / "netcdfs" / sub_path
    
    if not local_path.exists():
        # Fallback: download if not cached
        download_and_cache_netcdf(sub_path)
    
    with open(local_path, "rb") as f:
        return f.read()

@mcp.resource("lakefs://csv/{sub_path}")
def get_csv_binary(sub_path: str, context: Context):
    """
    Returns binary content of a CSV file from persistent storage.
    """
    local_path = DATA_DIR / "csv" / sub_path
    
    if not local_path.exists():
        # Fallback: download if not cached
        download_and_cache_csv(sub_path)
    
    with open(local_path, "rb") as f:
        return f.read()

# --- Tools --- #

@mcp.tool()
async def show_available_datasets(data_type: str, context: Context) -> str:
    """
    Generates a tree-like string showing available dataset paths.

    Parameters
    ----------
    data_type : str
        The file type.
        Supported options are: 'csv', 'shapefile', 'netcdf'

    Returns:
    --------
    str
        Tree structure of available datasets.
    """
    paths = get_available_paths(data_type)

    if not paths:
        raise ValueError(f"No {data_type} files found.")

    tree = {}
    for path in paths:
        parts = path.split('/')
        current = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current[part] = None
            else:
                current = current.setdefault(part, {})

    lines = []

    def print_tree(node, prefix=""):
        items = list(node.items())
        for i, (name, children) in enumerate(items):
            is_last = (i == len(items) - 1)
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "

            if children is None:
                lines.append(f"{prefix}{current_prefix}{name}")
            else:
                lines.append(f"{prefix}{current_prefix}{name}/")
                print_tree(children, prefix + next_prefix)

    for i, (name, children) in enumerate(tree.items()):
        if children is None:
            lines.append(f"{name}")
        else:
            lines.append(f"{name}/")
            print_tree(children, prefix="")

    icon = {
        "csv": "📊",
        "shapefile": "🗺️",
        "netcdf": "🌐"
    }.get(data_type, "📄")

    lines.append(f"\n{icon} Total {data_type} datasets: {len(paths)}")

    return "\n".join(lines)

@mcp.tool()
async def query_shapefile(
    shapefile_path: str,
    polygon_coords: list[dict],
    context: Context
    ) -> str:
    """
    Retrieves records from a geospatial dataset (shapefile)
    whose geometries intersect a given polygon.
    To explore available shapefiles, use `show_available_datasets('shapefile')` beforehand.

    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile to query.
    polygon_coords : list of dict
        List of coordinates pairs in {"lat": float, "lng": float} format
        representing the vertices of the polygon.

    Returns:
    --------
    str
        Plain-text summary of matching records.
    """

    from utils import parse_coordinates, make_gdf_summary
    
    # Load Shapefile
    gdf = load_shapefile_to_gdf(shapefile_path)
    
    # Define polygon
    polygon_coords = parse_coordinates(polygon_coords)
    polygon = Polygon(polygon_coords)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    # Filter rows that intersect the polygon
    matches = gdf[gdf.geometry.intersects(polygon)]
    
    if matches.empty:
        raise ValueError("No records were found within the specified polygon.")

    return make_gdf_summary(matches)

@mcp.tool()
async def query_netcdf(
    netcdf_path: str,
    polygon_coords: list[dict],
    agg_func: str,
    context: Context
    ) -> dict:
    """
    Aggregates records from a geospatial dataset (netcdf)
    whose geometries intersect a given polygon.
    To explore available netcdf files, use `show_available_datasets('netcdf')` beforehand.

    Parameters
    ----------
    netcdf_path : str
        Path to the NetCDF file.
    polygon_coords : list of dict
        List of coordinates pairs in {"lat": float, "lng": float} format
        representing the vertices of the polygon.
    agg_func : str
        Aggregation function to apply.
        Supported options are: 'mean', 'median', 'sum', 'std', 'min', 'max', or 'series'.

    Returns
    -------
    dict
        Dict containing the aggregated value of the variable within the polygon.
    """

    from utils import parse_coordinates
    
    # Supported aggregation functions
    allowed_funcs = ['mean', 'median', 'sum', 'std', 'min', 'max', 'series']
    if agg_func not in allowed_funcs:
        raise ValueError(
            f"Aggregation function '{agg_func}' is not supported.\n"
            f"Supported options are: {allowed_funcs}."
        )

    # Load NetCDF
    nc_data = load_netcdf_data(netcdf_path)
    gdf = nc_data.get("gdf")

    # Define polygon
    polygon = Polygon(parse_coordinates(polygon_coords))
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    # Select points within polygon
    gdf['in_polygon'] = gdf.geometry.within(polygon)
    inside_indices = gdf[gdf['in_polygon']].index
    
    if inside_indices.empty:
        raise ValueError("No satellite data found within the specified polygon.")
    
    # Get values
    values = nc_data.get("values")
    filtered_values = values.flatten()[inside_indices]

    output = {
        "description": nc_data.get("description"),
        "start_time": nc_data.get("start_time"),
        "end_time": nc_data.get("end_time"),
        "units": nc_data.get("units"),
    }

    if agg_func == 'series':
        output["series"] = filtered_values.tolist()
    else:
        result = getattr(np, "nan" + agg_func)(filtered_values)
        output[agg_func] = round(float(result),2)

    # NaN summary
    nan_count = int(np.isnan(filtered_values).sum())
    if nan_count:
        output["nan_warning"] = f"{nan_count} NaN values out of {len(filtered_values)} records"

    return output

@mcp.tool()
async def get_csv_schema(csv_path: str, context={}) -> str:
    """
    Return the schema (columns and dtypes) for the CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the csv to explore.

    Returns
    -------
    str
        A description of the fields in the CSV file, including sample values and format info.
    """

    # Load CSV
    df = load_csv_to_df(csv_path)

    table_name = Path(csv_path).stem
    column_info = []

    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"

        format_hint = ""
        if dtype == object:
            if isinstance(sample, str):
                if sample.isupper():
                    format_hint = "UPPERCASE"
                elif sample.islower():
                    format_hint = "lowercase"
        elif "datetime" in str(dtype):
            format_hint = "format: YYYY-MM-DD"
        elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            pass  # no format info needed
        elif pd.api.types.is_object_dtype(dtype) and isinstance(sample, str):
            try:
                pd.to_datetime(sample)
                format_hint = "possibly a date (check format)"
            except Exception:
                pass

        format_display = f", {format_hint}" if format_hint else ""
        column_info.append(f"{col} ({dtype}, e.g. '{sample}'{format_display})")

    return f"{table_name}:\n  - " + "\n  - ".join(column_info)

@mcp.tool()
async def query_csv(csv_path: str, sql_query: str, context: Context) -> str:
    """
    Executes an SQL query on a CSV using SQLite syntax and returns up to 10 rows of query results.
    To explore available csv files, use `show_available_datasets('csv')` beforehand.

    Parameters
    ----------
    csv_path : str
        Path to the csv to query.
    sql_query : str
        The SQL query to execute.
        The FROM clause should reference the table name, which is the stem of the csv_path.

    Returns:
    --------
    str
        The first 10 rows of the query result formatted as a CSV string.
        If more rows exist, a note is added at the end indicating how many rows were not shown.
    """

    from utils import enforce_limit, create_count_query

    # Load CSV
    df = load_csv_to_df(csv_path)
    table_name = Path(csv_path).stem
    database = {table_name: df}

    # Calculate total count efficiently using COUNT(*)
    count_query = create_count_query(sql_query)
    count_result = ps.sqldf(count_query, database)
    total_count = count_result.iloc[0, 0]
    
    # Execute query with enforced LIMIT
    parsed_query = enforce_limit(sql_query)
    query_result = ps.sqldf(parsed_query, database)
    
    if query_result.empty:
        raise ValueError("The query returned no results.")
    
    # Format query output as CSV string
    returned_count = len(query_result)
    query_result = query_result.to_csv(index=False)
    
    if total_count > returned_count:
        query_result += f"\n... ({total_count - returned_count} more rows not shown)"
    
    return query_result

@mcp.tool()
async def calculate_geodesic_area(
    polygon_coords: list[dict],
    context: Context) -> str:
    """
    Calculate the geodesic area of a polygon.

    Parameters
    ----------
    polygon_coords : list of dict
        List of coordinates pairs in {"lat": float, "lng": float} format
        representing the vertices of the polygon.

    Returns
    -------
    str
        Area in m² or km².
    """

    from utils import parse_coordinates, format_metric

    geod = Geod(ellps="WGS84")

    # Define polygon
    polygon_coords = parse_coordinates(polygon_coords)
    polygon = Polygon(polygon_coords)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    area, _ = geod.geometry_area_perimeter(polygon)
    return format_metric(abs(area), unit="m2")

@mcp.tool()
async def calculate_geodesic_perimeter(
    polygon_coords: list[dict],
    context: Context) -> str:
    """
    Calculate the geodesic perimeter of a polygon.

    Parameters
    ----------
    polygon_coords : list of dict
        List of coordinates pairs in {"lat": float, "lng": float} format
        representing the vertices of the polygon.

    Returns
    -------
    str
        Perimeter in m or km.
    """

    from utils import parse_coordinates, format_metric
    
    geod = Geod(ellps="WGS84")

    # Define polygon
    polygon_coords = parse_coordinates(polygon_coords)
    polygon = Polygon(polygon_coords)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    _, perimeter = geod.geometry_area_perimeter(polygon)
    return format_metric(perimeter, unit="m")

# --- Run MCP --- #

if __name__ == "__main__":
    # Initialize data cache before starting the server
    asyncio.run(initialize_data())
    
    mcp.run(
        #transport="streamable-http",
        #host="0.0.0.0",
        #port=8000,
        #path="/mcp",
        #log_level="info",
    )