# test.py

import asyncio
from fastmcp import Client

client = Client("http://0.0.0.0:8000/mcp") # src/main.py

async def test_tools(polygon_coordinates):
    async with client:
        result = await client.call_tool("mpa_feasibility_report", {"polygon_coordinates": polygon_coordinates})
        print(result.data)

# Polygon coordinatess for Isla Espíritu Santo, Baja California, México
polygon_coordinates = [
    {"lat": 24.664169, "lng": -110.209952},
    {"lat": 24.359598, "lng": -110.209952},
    {"lat": 24.359598, "lng": -110.489539},
    {"lat": 24.664169, "lng": -110.489539},
    {"lat": 24.664169, "lng": -110.209952}
]

asyncio.run(test_tools(polygon_coordinates=polygon_coordinates))