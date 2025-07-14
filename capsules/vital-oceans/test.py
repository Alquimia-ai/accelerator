import asyncio
from fastmcp import Client
from pprint import pprint

client = Client("main.py") # http://0.0.0.0:8000/mcp 

async def test_tools(coords):
    async with client:

        results = []

        print("\n\n--- Shapefile Data ---")
        result = await client.call_tool("show_available_paths", {"file_type": "shapefile"})
        pprint(result[0].text)

        print("- Corales")
        result = await client.call_tool("get_shapefile_summary", {"path": "habitats/corales/warmwatercoralreef_extent.shp", "coords": coords})
        pprint(result[0].text)

        print("- AMP (Mexico)")
        result = await client.call_tool("get_shapefile_summary", {"path": "areas_prioritarias/areas_protegidas/mexico/anp-mexico.shp", "coords": coords})
        pprint(result[0].text)

        print("- AMP (EEUU)")
        result = await client.call_tool("get_shapefile_summary", {"path": "areas_prioritarias/areas_protegidas/america_norte/PCA_Baja_to_Bering_2005.shp", "coords": coords})
        pprint(result[0].text)

        print("- Biodiversidad:")
        result = await client.call_tool("get_shapefile_summary", {"path": "areas_prioritarias/zonas_importancia_biodiversidad/zona_importancia_Biodiv_CDB.shp", "coords": coords})
        pprint(result[0].text)

        print("- Biología:")
        result = await client.call_tool("get_shapefile_summary", {"path": "areas_prioritarias/zonas_importancia_biológica/zona_importancia_biol_pp.shp", "coords": coords})
        pprint(result[0].text)

        print("- Refugios Pesqueros:")
        result = await client.call_tool("get_shapefile_summary", {"path": "areas_prioritarias/zonas_refugios_pesqueros/zona_refugio_pesquero_COBI2020.shp", "coords": coords})
        pprint(result[0].text)

        print("- Turismo:")
        result = await client.call_tool("get_shapefile_summary", {"path": "socioeconomico/turismo/INEGI_DENUE_26052025.shp", "coords": coords})
        pprint(result[0].text)

        print("- Pesca:")
        result = await client.call_tool("get_shapefile_summary", {"path": "socioeconomico/pesca/INEGI_DENUE_26052025.shp", "coords": coords})
        pprint(result[0].text)

        print("- Zonas explotación (calamar):")
        result = await client.call_tool("get_shapefile_summary", {"path": "socioeconomico/zonas_explotacion_pesquera/calamar/zona_pesca_calamar.shp", "coords": coords})
        pprint(result[0].text)

        print("\n\n--- NetCDF Data ---")
        result = await client.call_tool("show_available_paths", {"file_type": "netcdf"})
        pprint(result[0].text)

        print("- Chlorophylla:")
        result = await client.call_tool("get_netcdf_stats", {"path": "chlorophylla.nc", "coords": coords, "mode": "summary"})
        pprint(result[0].text)

        print("- Temperature:")
        result = await client.call_tool("get_netcdf_stats", {"path": "temperature.nc", "coords": coords, "mode": "summary"})
        pprint(result[0].text)

        print("- Temperature:")
        result = await client.call_tool("get_netcdf_stats", {"path": "temperature.nc", "coords": coords, "mode": "series"})
        pprint(result[0].text)

        print("\n\n--- CSV Data ---")
        result = await client.call_tool("show_available_paths", {"file_type": "csv"})
        pprint(result[0].text)

        print("- Schema:")
        result = await client.call_tool("get_csv_schema", {"path": "tablas_biodiversidad/peces_bahia_lapaz.csv"})
        pprint(result[0].text)

        print("- Query:")
        result = await client.call_tool("query_csv", {"path": "tablas_biodiversidad/peces_bahia_lapaz.csv", "sql_query": "SELECT * FROM peces_bahia_lapaz"})
        pprint(result[0].text)

        print("\n\n--- Geodesic ---")
        result = await client.call_tool("calculate_geodesic_perimeter", {"coords": coords})
        print(f"- Perimeter: {result[0].text}")
        
        result = await client.call_tool("calculate_geodesic_area", {"coords": coords})
        print(f"- Area: {result[0].text}")

# Polygon coords from Baja California
coords = [
    {
        "lat": 25.65,
        "lng": -117.29
    },
    {
        "lat": 26.63,
        "lng": -110.03
    },
    {
        "lat": 22.32,
        "lng": -106.43
    },
    {
        "lat": 20.11,
        "lng": -108.80
    }
]

asyncio.run(test_tools(coords=coords))