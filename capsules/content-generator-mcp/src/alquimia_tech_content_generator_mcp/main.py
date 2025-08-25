import json
import mimetypes
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel

from alquimia_tech_content_generator_mcp.client import AlquimiaClient

mcp = FastMCP("Technical content generator MCP")

API_KEY = os.environ.get("API_KEY", None)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
CONTENT_GENERATOR_AGENT_ID = os.environ.get(
    "CONTENT_GENERATOR_AGENT_ID", "code_reviewer"
)
TECHNICAL_AGENT_ID = os.environ.get("TECHNICAL_AGENT_ID", "technical_assistant")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "chat")
MAX_INPUT_BATCH_SIZE = int(os.environ.get("MAX_INPUT_BATCH_SIZE", 8000))

SESSION_ID = uuid.uuid4()

client = AlquimiaClient(
    BASE_URL,
    str(SESSION_ID),
    CONTENT_GENERATOR_AGENT_ID,
    CHANNEL_ID,
    api_key=API_KEY,
)


class Subtopic(BaseModel):
    title: str
    files: List[str]


class Topic(BaseModel):
    title: str
    files: List[str]
    subtopics: Optional[List[Subtopic]] = []


class DocumentationStructure(BaseModel):
    topics: List[Topic]


async def calculate_diff(files: list[str]) -> str:
    """
    Calculate git diff based on given project's files
    """
    # Convert to relative paths (required by git)
    repo_root = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    relative_paths = [os.path.relpath(path, repo_root) for path in files]

    # Run git diff on those files
    diff = subprocess.check_output(
        ["git", "diff", "--"] + relative_paths, cwd=repo_root
    ).decode()
    return diff


async def attach_context(files: list[str]) -> str:
    """
    Attach the given set of files into the context
    """
    attachments = []
    for path in filter(os.path.isfile, files):
        content_type, _ = mimetypes.guess_type(path)
        size = os.path.getsize(path)

        attachments.append(
            {
                "content_type": content_type,
                "content_size": size,
                "filename": os.path.basename(path),
            }
        )

    response = await client.infer("1", attachments=attachments)
    attachment_ids = response["attachments"]
    stream_id = response["stream_id"]

    for idx, path in enumerate(filter(os.path.isfile, files)):
        await client.upload_attachment(stream_id, SESSION_ID, attachment_ids[idx], path)

    # wait for attachments to complete
    response = await client.stream(stream_id)
    logger.debug("Attachment response: %s" % response)

    return "Done!"


@mcp.tool
async def create_entire_wiki(
    project_files: List[str],
    theme: str,
    project_insights: str,
    project_docs_root_path: str,
) -> List[str]:
    """
    Generate a complete wiki-style documentation set for a project.

    This tool analyzes the provided project files, organizes them into
    thematic topics and subtopics, and generates human-readable markdown
    documentation around the given **theme** (the central point of interest).
    The generated documents are stored in the specified output folder.

    Args:
        project_files (List[str]):
            A list of absolute or relative source code file paths that belong to the project.
            These files are analyzed and used to build technical documentation.
        theme (str):
            The main theme or focus area around which the wiki will be structured
            (e.g., "Data Pipeline", "Authentication System").
        project_insights (str):
            What the project is about. Include relevant data as the project usage (i.e.
            lib, service, etc) and a brief description.
        project_docs_root_path (str):
            Path to the root folder where the generated documentation files
            will be stored. The function ensures this directory exists.

    Returns:
        List[str]:
            A list of file paths (strings) corresponding to the generated
            wiki markdown documents.

    Side Effects:
        - Creates `.md` files in the specified `project_docs_root_path`.
        - Each topic and its subtopics are written as separate files.

    """
    wiki_docs: List[str] = []

    await attach_context(project_files)

    # Ensure root path exists
    root_path = Path(project_docs_root_path)
    root_path.mkdir(exist_ok=True)

    # Validate structure with Pydantic
    raw_structure = await create_document_structure(
        project_files, theme, project_insights
    )

    structure = DocumentationStructure(**raw_structure)

    for idx, topic in enumerate(structure.topics):
        # Main topic content
        page_content = await create_technical_topic_from_files(
            topic.files, f"{theme}: {topic.title}"
        )

        # Save file
        doc_name = root_path / f"{idx}-{topic.title.replace(' ', '_')}.md"
        with open(doc_name, "w", encoding="utf-8") as f:
            f.write(page_content)

        wiki_docs.append(str(doc_name))

    index = await create_technical_topic_from_files(
        wiki_docs, f"{theme}: Main index & introduction"
    )
    doc_name = root_path / "README.md"
    with open(doc_name, "w", encoding="utf-8") as f:
        f.write(index)

    wiki_docs.append(str(doc_name))
    return wiki_docs


def extract_brace_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or start >= end:
        return

    return text[start : end + 1]


async def create_document_structure(
    files: list[str], theme: str, project_insights: str
) -> dict:
    """
    Create a wiki project structure to cover the given project structure
    """
    _client = AlquimiaClient(
        BASE_URL,
        str(SESSION_ID),
        TECHNICAL_AGENT_ID,
        CHANNEL_ID,
        api_key=API_KEY,
    )

    response = await _client.infer(f"""
        Imagine an ideal  **wiki-style documentation structure** for the given project based on the following structure: {map_files_with_metadata(files)}.
        The main focus of interest is: `{theme}`.
        Project insights: `{project_insights}`

        Requirements:
        - Respond **only in strict JSON format**.
        - Output must be a hierarchical tree of "topics".
        - Each "topic" must reference **at least 3 and no more than 5 files**.
        - Don't repeat the same source code file more than three times between topics.
        - Ensure all project files are covered in a topic
        - Consider file sizes and relevance when agrouping topics
        - Group topics logically to highlight architecture, dependencies, modules, services, integrations, and utilities based on the main foucs of interest.
        - Each node must contain:
            - "title": a short descriptive string
            - "files": a list of file paths

        Example JSON schema:
        {{
          "topics": [{{
              "title": "Architecture",
              "files": ["file1.py", "file2.py", "file3.py"]
          }}]
        }}
        Respond only with JSON with no extra characters
    """)

    stream_id = response["stream_id"]
    content = await _client.stream(stream_id)
    adict = extract_brace_block(content)
    try:
        return json.loads(adict)
    except:
        logger.warn("Parsing failed! Trying replacing quotes..")
        return json.loads(adict.replace("'", '"'))


async def create_technical_topic_from_files(files: list[str], topic: str) -> str:
    """
    Create technical documentation based on the given set of relevant source code files and topic.
    """

    chunks = []
    for f in files:
        with open(f, "r", encoding="utf-8") as _f:
            chunks.extend([_f.read()])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_INPUT_BATCH_SIZE,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    splitted = splitter.create_documents(chunks)

    content = ""
    for idx, chunk in enumerate(splitted):
        logger.debug(f"{topic}: PROCESSING CHUNK {idx}/{len(splitted)}")
        response = await client.infer(
            str(chunk),
            extra_data={
                "current_doc_state": content,
                "relevant_files": files,
                "doc_topic": topic,
            },
        )
        stream_id = response["stream_id"]
        content = await client.stream(stream_id)
        time.sleep(2)

    return content


async def create_release_notes_from_files(files: list[str]) -> str:
    """
    Create a release notes based on a git diff output
    """
    CLAUSES = [
        "Generate a release notes based on the given git diff's output",
        "If the current release notes state is shared, rewrite as needed without losing the original meaning",
        "Do not share false or misguiding information",
    ]

    diff = await calculate_diff(files)
    chunked_diff = split_text_by_chars(diff)
    total_chunks = len(chunked_diff)
    response = await client.infer(
        "\n".join(
            CLAUSES + [f"Git diff's output: 1 of {total_chunks}: {chunked_diff[0]}"]
        )
    )
    stream_id = response["stream_id"]
    release_notes = await client.stream(stream_id)
    if len(chunked_diff) > 1:
        for idx, chunk in enumerate(chunked_diff[1:]):
            response = await client.infer(
                "\n".join(
                    CLAUSES
                    + [
                        f"Current release notes: {release_notes}",
                        f"nGit diff {idx} of {total_chunks}: {chunk}",
                    ]
                )
            )
            stream_id = response["stream_id"]
            release_notes = await client.stream(stream_id)

    return release_notes


def map_files_with_metadata(files: list[str]) -> list[dict]:
    file_metadata = []
    for f in files:
        try:
            stats = os.stat(f)
            file_metadata.append(
                {
                    "path": f,
                    "size_bytes": stats.st_size,
                    "size_kb": round(stats.st_size / 1024, 2),
                    "modified_time": stats.st_mtime,
                    "created_time": stats.st_ctime,
                }
            )
        except FileNotFoundError:
            file_metadata.append({"path": f, "error": "File not found"})
    return file_metadata


def split_text_by_chars(text: str, max_chars: int = MAX_INPUT_BATCH_SIZE) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    mcp.run()
