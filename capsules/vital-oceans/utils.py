# utils.py

from typing import Union, Literal, List

import os
import json
import csv
from io import StringIO
from pathlib import Path
from math import sqrt, pi
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from pandasql import sqldf

import shapely
from shapely import MultiPolygon, MultiPoint, Polygon, Point

import lakefs

#from pyproj import Geod, CRS
#from shapely.geometry import MultiPolygon, MultiPoint, Polygon, Point

def get_shp_paths(dir: str, ref: lakefs.Repository.ref) -> List[str]:
    """
    List all .shp files under a given directory path in the lakeFS repo.
    Returns relative paths (within that directory).
    """
    base = "shapefiles/"
    prefix = base + dir.rstrip('/') + '/'
    shp_paths: List[str] = []

    # Iterate over objects in the specified ref
    for obj in ref.objects(prefix=prefix):
        if obj.path.endswith(".shp"):
            rel = obj.path[len(base):]
            shp_paths.append(rel)

    return shp_paths


def parse_coordinates(coordinates: list[dict]) -> list[tuple]:
    """Converts a list of {'lat', 'lng'} dicts to (lng, lat) tuples."""
    return [(pair["lng"], pair["lat"]) for pair in coordinates]


def calculate_geodesic_area(polygon: shapely.Polygon) -> float:
    """Calculate the geodesic area of a polygon."""

    # From polygon to gdf
    gdf_polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs="EPSG:4326")
    gdf_polygon = gdf_polygon[gdf_polygon.geometry.notnull()]
    
    # Project CRS for area calculation
    epsg = 32612 # PCS: WGS 1984 UTM Zone 12N
    gdf_polygon = gdf_polygon.to_crs(epsg=epsg)

    # Compute area in km2
    area_km2 = (gdf_polygon.area / 1e6).iloc[0]
    
    return round(area_km2,2)


def csv_to_md(csv_string: str, max_rows: int = 50) -> str:
    """Convert a CSV string to a Markdown table"""
    reader = csv.reader(StringIO(csv_string.strip()))
    rows = list(reader)
    if not rows:
        return ""

    header = rows[0]
    separator = ['---'] * len(header)
    body = rows[1:max_rows + 1]

    table = [header, separator] + body

    truncated = len(rows) - 1 > max_rows
    if truncated:
        table.append(['...'] * len(header))

    md_lines = ['| ' + ' | '.join(row) + ' |' for row in table]
    if truncated:
        md_lines.append(f"({len(rows) - 1 - max_rows} rows not shown)")

    return '\n'.join(md_lines)


def count_geoms_in_gdf(gdf: gpd.GeoDataFrame) -> tuple[int]:
    """..."""
    polygons, points = 0, 0
    for geom in gdf.geometry:
        if isinstance(geom, MultiPolygon):
            polygons += len(geom.geoms)
        elif isinstance(geom, Polygon):
            polygons += 1
        elif isinstance(geom, MultiPoint):
            points += 1
        elif isinstance(geom, Point):
            points += 1

    return polygons, points


def filter_points_within(
    gdf_shp: gpd.GeoDataFrame,
    polygon_clip: shapely.Polygon,
    buffer_km: float = 0.0,
    epsg: int = 32612  # Projected CRS for buffering
) -> gpd.GeoDataFrame:
    """
    Returns only the valid points from gdf_shp that fall within the (optionally buffered) polygon_clip.

    Parameters:
        - gdf_shp: GeoDataFrame containing points
        - polygon_clip: Polygon to use for filtering
        - buffer_km: Optional buffer (in km) to expand the clip polygon before filtering
        - epsg: EPSG code for buffering (default UTM 32612)
    """
    if not polygon_clip.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    # Filter invalid geometries
    gdf_shp = gdf_shp[gdf_shp.geometry.notnull() & gdf_shp.geometry.is_valid]

    # Ensure CRS is set
    if gdf_shp.crs is None:
        raise ValueError("GeoDataFrame must have a CRS defined.")

    # Project polygon to buffering CRS
    polygon_gdf = gpd.GeoSeries([polygon_clip], crs=gdf_shp.crs).to_crs(epsg=epsg)
    buffered_polygon = polygon_gdf.iloc[0]

    if buffer_km > 0:
        buffer_radius_m = buffer_km * 1000  # convert km to meters
        buffered_polygon = buffered_polygon.buffer(buffer_radius_m)

    # Reproject buffered polygon back to original CRS
    buffered_polygon = gpd.GeoSeries([buffered_polygon], crs=f"EPSG:{epsg}").to_crs(gdf_shp.crs).iloc[0]

    # Filter points within polygon
    filtered = gdf_shp[gdf_shp.geometry.within(buffered_polygon)]

    return filtered


def clip_and_calculate_overlap(
    gdf_shp: gpd.GeoDataFrame,
    gdf_clip: gpd.GeoDataFrame,
    buffer_km: float = 0.0,  # Buffer distance in km to expand the clip polygon
    epsg: int = 32612  # Projected CRS for area calculation
) -> gpd.GeoDataFrame:
    """
    Clips gdf_shp to the area of gdf_clip (optionally buffered) and calculates
    overlap area and ratio in km². Reprojects to the given EPSG code for accurate area calculations.

    Parameters:
        - gdf_shp: GeoDataFrame to be clipped
        - gdf_clip: Clipping GeoDataFrame
        - epsg: EPSG code for projection (default 32612)
        - buffer_km: Optional buffer (in km) to expand the clip polygon before clipping
    """
    gdf_shp = gdf_shp[gdf_shp.geometry.notnull()]
    gdf_clip = gdf_clip[gdf_clip.geometry.notnull()]

    if gdf_shp.crs is None:
        raise ValueError("Shapefile has no CRS defined.")
    if gdf_shp.crs != gdf_clip.crs:
        gdf_clip = gdf_clip.to_crs(gdf_shp.crs)

    # Union of clip geometries
    clip_union = gdf_clip.union_all()

    # Reproject union for buffering
    clip_union_proj = gpd.GeoSeries([clip_union], crs=gdf_shp.crs).to_crs(epsg=epsg).iloc[0]

    # Apply buffer if requested
    if buffer_km > 0:
        buffer_radius_m = buffer_km * 1000  # convert km to meters
        clip_union_proj = clip_union_proj.buffer(buffer_radius_m)

    # Back-project buffered clip to original CRS
    clip_union = gpd.GeoSeries([clip_union_proj], crs=f"EPSG:{epsg}").to_crs(gdf_shp.crs).iloc[0]

    # Clip input shapes
    clipped = gdf_shp[gdf_shp.geometry.intersects(clip_union)].copy()
    clipped["geometry"] = clipped.geometry.intersection(clip_union)

    # Reproject clipped for area calculation
    clipped = clipped.to_crs(epsg=epsg)
    clip_union_proj = gpd.GeoSeries([clip_union], crs=gdf_shp.crs).to_crs(epsg=epsg).iloc[0]
    clip_area_km2 = clip_union_proj.area / 1e6

    # Calculate overlap area and ratio
    clipped["overlap_area"] = (clipped.geometry.area / 1e6).round(2)
    clipped["overlap_ratio"] = (clipped["overlap_area"] / clip_area_km2).round(4)

    return clipped


def generate_report_from_gdf(
    shp_path: str,
    schema: dict,
    gdf: gpd.GeoDataFrame,
    polygon: shapely.Polygon,
    buffer_km: float = 0.0
) -> str:
    """Generate a Markdown-formatted report from gdf
    using geospatial filters (polygon clip) and SQL queries (shapefile schema)."""

    # Geospatial analysis
    geom_types = set(gdf.geometry.geom_type)
    if any(elem in geom_types for elem in ["Point", "MultiPoint"]):
        # Filter points within
        gdf = filter_points_within(
            gdf_shp=gdf,
            polygon_clip=polygon,
            buffer_km=buffer_km
        )
    elif any(elem in geom_types for elem in ["Polygon", "MultiPolygon"]):
        # Clip and calculate overlap
        gdf_clip = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs="EPSG:4326")
        gdf = clip_and_calculate_overlap(
            gdf_shp=gdf,
            gdf_clip=gdf_clip,
            buffer_km=buffer_km
        )
    else:
        return "" # NO geom type match

    # Count polygons/points
    polygons, points = count_geoms_in_gdf(gdf)

    # Run main sql query
    sql_queries = schema.get("sql_query")
    if not sql_queries:
        return "" # No sql queries
    
    if isinstance(sql_queries, str):
        sql_queries = [sql_queries] # Normalize to list of queries
    
    try:
        df_main = sqldf(sql_queries[0], {"gdf": gdf.drop(columns=["geometry"])}) 
    except Exception as e:
        raise ValueError(f"SQL Error in main query for {shp_path}:\n{sql_queries[0]}\nError: {e}")
    
    # Run additional sql queries if any
    extras = []
    for i, query in enumerate(sql_queries[1:], start=2):
        try:
            extra_df = sqldf(query, {"gdf": gdf.drop(columns=["geometry"])})
            if len(extra_df):
                extras.append((query, extra_df))
        except Exception as e:
            raise ValueError(f"SQL Error in extra query {i} for {shp_path}:\n{query}\nError: {e}")
            
    # Write report
    if len(df_main):
        csv = df_main.to_csv(index=False)
        md_table = csv_to_md(csv)
    
        multigeom_count = (
            f" -> {polygons} polygons" if "MultiPolygon" in geom_types
            else f" -> {points} points" if "MultiPoint" in geom_types
            else ""
        )
    
        report = f"""
## **{schema.get("title", shp_path)}**
{md_table}

- **Description**: {schema.get("description", "No description available")}
- **Total records**: {len(df_main)}
- **Geometry type(s)**: {', '.join(geom_types)}{multigeom_count}
- **Data Source**: {schema.get("source", "unknown")}
"""

        if extras:
            report += "**Additional insights**:"

            for query, extra_df in extras:
                extra_csv = extra_df.to_csv(index=False)
                extra_md = csv_to_md(extra_csv)
                report += f"\n{extra_md}\n"

    else:
        report = f"""
## **{schema.get("title", shp_path)}**
No records found.
"""

    report += "-"*10+"\n" 
    return report