from pathlib import Path
import tempfile
import zipfile
import asyncio
import os
import io

import numpy as np
import pandas as pd
import geopandas as gpd
import mo_sql_parsing as mosp

def parse_coordinates(coordinates):
    return [(pair["lng"], pair["lat"]) for pair in coordinates]

def make_gdf_summary(gdf: gpd.GeoDataFrame, schema: dict, top_n: int) -> str:
    """
    Generates a summary of a GeoDataFrame based on a provided schema.
    """
    summary_parts = ["🔍 **Summary:**"]
    total = len(gdf)
    summary_parts.append(f"- Total records found: {total}")

    for var_spec in schema.get("variables", []):
        var_name = var_spec.get("name")
        var_type = var_spec.get("type")

        if var_name not in gdf.columns:
            #print(f"⚠️ Col `{var_name}` not found.")
            continue

        series = gdf[var_name].dropna()

        if var_type == "text":
            top_vals = series.astype(str).value_counts().head(top_n).to_dict()
            summary_parts.append(f"- Categories in `{var_name}`: {top_vals}")

        elif var_type == "number":
            agg_func = var_spec.get("agg", "mean")
            if not pd.api.types.is_numeric_dtype(series):
                #print(f"⚠️ Col `{var_name}` is not a number.")
                continue

            try:
                agg_map = {
                    "count": series.count,
                    "mean": series.mean,
                    "sum": series.sum,
                    "min": series.min,
                    "max": series.max,
                    "std": series.std,
                }

                if agg_func not in agg_map:
                    #print(f"⚠️ `{agg_func}` is not a valid agg function for variable `{var_name}`")
                    continue

                result = agg_map[agg_func]()
                summary_parts.append(f"- `{agg_func}` of `{var_name}`: {result:.2f}" if isinstance(result, float) else f"- `{agg_func}` of `{var_name}`: {result}")

            except Exception as e:
                print(f"⚠️ Error computing `{agg_func}` for `{var_name}`: {e}")

    return "\n".join(summary_parts)

def format_metric(value: float, unit: str) -> str:
    if unit == "m":
        if value >= 1000:
            return f"{value / 1000:,.2f} km"
        return f"{value:,.2f} m"
    elif unit == "m2":
        if value >= 1_000_000:
            return f"{value / 1_000_000:,.2f} km²"
        return f"{value:,.2f} m²"
    else:
        return f"{value:,.2f} {unit}"

def enforce_limit(sql_query: str, max_limit=10) -> str:
    """
    Parses and rewrites a SQL query to enforce LIMIT constraints.
    """
    parsed = mosp.parse(sql_query)
    
    # Parse and enforce LIMIT
    user_limit = parsed.get("limit")
    
    if user_limit is None:
        limit = max_limit
    else:
        limit = (
            user_limit.get("value") if isinstance(user_limit, dict) else user_limit
        )
        limit = min(limit, max_limit)

    # Inject back the enforced LIMIT/OFFSET
    parsed["limit"] = limit

    return mosp.format(parsed)

def create_count_query(sql_query: str) -> str:
    """
    Converts a SQL query to a COUNT(*) query to get total row count efficiently.
    """
    parsed = mosp.parse(sql_query)
    
    # Wrap the original query as a subquery and count the results
    original_query = mosp.format(parsed)
    count_query = f"SELECT COUNT(*) FROM ({original_query}) AS subquery"
    
    return count_query
