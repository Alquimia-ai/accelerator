import json

import httpx
from aiosseclient import aiosseclient
from loguru import logger


class AlquimiaClient:
    """
    Alquimia HTTP client
    """

    def __init__(
        self,
        base_url,
        session_id,
        assistant_id,
        source,
        api_key=None,
        extra_data={},
        force_profile={},
        tools_load_module=None,
        event_stream_timeout=None,
        request_timeout=None,
        trigger_client_side_tools=True,
    ):
        self.session_id = session_id
        self.assistant_id = assistant_id
        self.source = source
        self.base_url = base_url
        self.api_key = api_key
        self.extra_data = extra_data
        self.force_profile = force_profile
        self.event_stream_timeout = event_stream_timeout
        self.request_timeout = request_timeout

    async def stream(self, stream_id):
        async for event in aiosseclient(
            f"{self.base_url}/stream/{stream_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_total=self.event_stream_timeout,
        ):
            try:
                data = json.loads(event.data)
            except ValueError:
                pass

            logger.debug(f"Event data: {data}")
            response = data.get("response", None)
            if response:
                return response["data"]["content"]

        logger.debug(f"Streaming ended STREAM_ID: {stream_id}")

    async def tool_completition(self, stream_id, tool_output):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/event/tool-completion",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"stream_id": stream_id, "tool_output": tool_output},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return response.json()

    async def upload_attachment(self, stream_id, session_id, attachment_id, path):
        logger.debug(f"Sending file as attachment {path}")
        with open(path, "rb") as _file:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/attachment/{session_id}/{stream_id}/{attachment_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=dict(file=_file),
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                return response.json()

    async def infer(self, query, attachments=[]):
        payload = {
            "query": query,
            "session_id": self.session_id,
            "extra_data": self.extra_data,
            "force_profile": self.force_profile,
            "attachments": attachments,
        }
        logger.debug(f"Sending inference request to Alquimia: {payload}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/infer/{self.source}/{self.assistant_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return response.json()
