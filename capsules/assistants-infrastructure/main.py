import httpx
from fastmcp import FastMCP
import os
from mcp.server.fastmcp.prompts import base
from transformers import PretrainedConfig
from dataclasses import dataclass
from typing import List, Dict, Any
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# Initialize FastMCP server
mcp = FastMCP("assistants-infraestructure")


mcp = FastMCP(
    "assistants-infraestructure",
    dependencies=[
        "httpx",
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "beautifulsoup4",
        "transformers",
    ],
)
HUGGINGFACE_ENDPOINT = "https://huggingface.co"
HUGGINGFACE_API_KEY = os.environ.get(
    "HUGGINGFACE_API_KEY", "your_huggingface_api_key_here"
)
DTYPE_SIZES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}
MODEL_TENSORS_BYTES = DTYPE_SIZES["fp16"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TWYD_API_KEY = os.environ.get("TWYD_API_KEY", "your_twyd_api_key_here")
TWYD_API_URL = os.environ.get(
    "TWYD_API_URL", "twyd-alquimia-twyd.apps.alquimiaai.hostmydemo.online"
)
TWYD_TOPIC_ID = os.environ.get("TWYD_TOPIC_ID", "42")

os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]


@mcp.tool()
async def get_specific_model_insights(model_name: str):
    """
    Fetches information about a Hugging Face model.

    Useful to get the model_id for further operations.

    Args:
        model_name (str): The name of the model to fetch information for.

    Returns:
        dict: A dictionary containing model information.
    """
    url = f"{HUGGINGFACE_ENDPOINT}/api/models?search={model_name}&sort=likes&limit=5&full=true&config=true"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        results = response.json()
        
        # Handle empty or invalid responses
        if not results or not isinstance(results, list):
            return [{"error": f"No model found with name '{model_name}'. Please check the model name and try again."}]
        
        return results


@mcp.tool()
async def get_models(leaderboard_type: str = "overall") -> str:
    """
    Reads and returns the leaderboard data from local markdown files.

    Args:
        leaderboard_type (str): The type of leaderboard to read.
                               Available options: 'overall', 'coding', 'english', 'math', 'spanish'.
                               Defaults to 'overall'.

    Returns:
        str: The contents of the specified leaderboard markdown file.
    """
    import os

    # Map leaderboard types to file names
    leaderboard_files = {
        "overall": "overall_leaderboard.md",
        "coding": "coding_leaderboard.md",
        "english": "english_leaderboard.md",
        "math": "math_leaderboard.md",
        "spanish": "spanish_leaderboard.md",
    }

    # Validate leaderboard type
    if leaderboard_type not in leaderboard_files:
        available_types = ", ".join(leaderboard_files.keys())
        raise ValueError(
            f"Invalid leaderboard_type '{leaderboard_type}'. Available options: {available_types}"
        )

    # Get the file path
    leaderboard_file = leaderboard_files[leaderboard_type]
    file_path = os.path.join("leaderboards", leaderboard_file)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Leaderboard file '{file_path}' not found.")

    # Read and return the file contents
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@mcp.tool()
async def get_hardware_requirements(
    model_id: str,
    model_size: int,
    desired_context_window: int = 20000,
):
    """
    Fetches the hardware requirements for a specific Hugging Face model.

    Args:
        model_id (str): The ID of the model to fetch hardware requirements for.
        model_size (int): The size of the model in billions of parameters.
        desired_context_window (int): The desired context window size for the model.

    Returns:
        dict: A dictionary containing hardware requirements.
    """

    try:
        config_dict = PretrainedConfig.from_pretrained(model_id).to_dict()
    except Exception as e:
        print("Error loading model configuration:", e)
        raise RuntimeError("Failed to load model config; cannot proceed.")

    hidden_size = (
        config_dict["text_config"]["hidden_size"]
        if "text_config" in config_dict
        else config_dict["hidden_size"]
    )

    hidden_layers = (
        config_dict["text_config"]["num_hidden_layers"]
        if "text_config" in config_dict
        else config_dict["num_hidden_layers"]
    )

    total_params = model_size * (10**9)
    taken_space_parameters = total_params * MODEL_TENSORS_BYTES
    rounded_taken_space_parameters = round(taken_space_parameters / 2**30, 3)
    kv_bytes_per_token = hidden_layers * (4 * hidden_size)
    desired_tokens = desired_context_window
    kv_required_bytes = desired_tokens * kv_bytes_per_token
    kv_required_gib = round(kv_required_bytes / 2**30, 3)

    total_needed_gib = round(rounded_taken_space_parameters + kv_required_gib, 3)

    return {
        "model_id": model_id,
        "model_size_params": model_size,
        "context_window": desired_context_window,
        "parameters_gib": rounded_taken_space_parameters,
        "kv_cache_gib": kv_required_gib,
        "total_gib": total_needed_gib,
        "hidden_size": hidden_size,
        "hidden_layers": hidden_layers
    }


@mcp.tool()
async def get_insights_use_case(query: str) -> List[Dict[str, Any]]:
    """
    Provides insights on a use case based on the query.

    Args:
        query (str): The query to analyze for insights.

    Returns:
        list[base.Message]: A list of messages containing insights.
    """
    payload = {"query": query, "k": 2}
    headers = {
        "Authorization": f"Bearer {TWYD_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"https://{TWYD_API_URL}/api/topics/{TWYD_TOPIC_ID}/search",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        results = resp.json()

    insights: List[Dict[str, Any]] = []
    
    # Handle empty or invalid responses
    if not results or not isinstance(results, list):
        return [{
            "pageContent": f"No insights found for query: '{query}'. Please try a different search term or check if the query is specific enough.",
            "metadata": {"status": "no_results", "query": query}
        }]
    
    for item in results:
        page_content = item.get("pageContent", "")
        metadata = item.get("metadata", {})
        insights.append({"pageContent": page_content, "metadata": metadata})
    
    # Return a helpful message if no insights were found
    if not insights:
        return [{
            "pageContent": f"No insights found for query: '{query}'. Please try a different search term or check if the query is specific enough.",
            "metadata": {"status": "no_results", "query": query}
        }]
    
    return insights


@mcp.tool()
async def get_runtime_consumption(tags: List[str]) -> List[Dict[str, Any]]:
    """
    Check for a given set of tags the runtime consumption with similar assistants that were previously run.
    Args:
        tags (List[str]): A list of tags to filter by. Returns entries that contain any of these tags.

    Returns:
        List[Dict[str, Any]]: A list of matching specifications from the runtime-consumption.json file.
    """
    import json
    import os

    json_file_path = "docs/runtime-consumption.json"

    if not os.path.exists(json_file_path):
        return []

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    stress_test_data = data.get("stress_test_resource_usage", [])
    matching_specs = []

    for spec in stress_test_data:
        spec_tags = spec.get("type", [])
        if any(tag.upper() in [t.upper() for t in spec_tags] for tag in tags):
            matching_specs.append(spec)

    return matching_specs


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        log_level="info",
    )
