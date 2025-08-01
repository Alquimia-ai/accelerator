# test.py

from pprint import pprint
import asyncio

from fastmcp import Client

client = Client("http://0.0.0.0:8000/mcp") # main.py

async def test_tools(polygon_coordinates):
    async with client:
        result = await client.call_tool("calculate_polygon_area", {"polygon_coordinates": polygon_coordinates})
        pprint(result[0].text)

        result = await client.call_tool("get_red_list_in_polygon", {"polygon_coordinates": polygon_coordinates, "data_source": "IUCN"})
        pprint(result[0].text)

        result = await client.call_tool("report_key_area_overlap", {"polygon_coordinates": polygon_coordinates, "buffer_km": 30})
        pprint(result[0].text)

        result = await client.call_tool("report_habitat_overlap", {"polygon_coordinates": polygon_coordinates})
        pprint(result[0].text)

        result = await client.call_tool("report_exploitation_area_overlap", {"polygon_coordinates": polygon_coordinates})
        pprint(result[0].text)

        result = await client.call_tool("find_nearby_coastal_communities", {"polygon_coordinates": polygon_coordinates, "buffer_km": 30})
        pprint(result[0].text)

        result = await client.call_tool("get_human_activity_in_polygon", {"polygon_coordinates": polygon_coordinates})
        pprint(result[0].text)

        result = await client.call_tool("calculate_social_lag_in_polygon", {"polygon_coordinates": polygon_coordinates})
        pprint(result[0].text)

        result = await client.call_tool("get_fauna_in_polygon", {"polygon_coordinates": polygon_coordinates})
        pprint(result[0].text)

# Polygon coordinatess from Isla Espíritu Santo, Baja California, México
polygon_coordinates = [
    {"lat": 24.664169, "lng": -110.209952},
    {"lat": 24.359598, "lng": -110.209952},
    {"lat": 24.359598, "lng": -110.489539},
    {"lat": 24.664169, "lng": -110.489539},
    {"lat": 24.664169, "lng": -110.209952}
]

asyncio.run(test_tools(polygon_coordinates=polygon_coordinates))