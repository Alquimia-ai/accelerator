import fnmatch
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional

import magic
from fastmcp import FastMCP
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel

from alquimia_tech_content_generator_mcp.client import AlquimiaClient

mcp = FastMCP("Technical content generator MCP")

API_KEY = os.environ.get("API_KEY", None)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/app/")
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
        # Detect MIME type from file content
        mime = magic.Magic(mime=True)
        content_type = mime.from_file(path)

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

    return "Done!"


@mcp.tool
async def create_documentation(
    source_roots: List[str],
    include_patterns: List[str],
    exclude_patterns: Optional[List[str]] = None,
    theme: str = "",
    project_insights: str = "",
    project_docs_root_path: str = ".",
) -> List[str]:
    """
    Generate a complete wiki-style documentation set for a project.

    Args:
        source_roots (List[str]):
            List of root directories to search for source files.
            Example: ["src/main/java", "application/src/main/java", "webui/persistence/src/main/java"]
        include_patterns (List[str]):
            File patterns to include (e.g., ["**/*.java", "**/*.xml"]).
        exclude_patterns (List[str], optional):
            File patterns to exclude (e.g., ["**/target/**", "**/*.class"]).
        theme (str):
            The main theme or focus area for the wiki (e.g., "System Design").
        project_insights (str):
            General description/insights about the project.

    Returns:
        List[str]: List of generated markdown file paths.
    """
    root_path = Path(PROJECT_ROOT)
    root_path.mkdir(parents=True, exist_ok=True)

    all_files: List[Path] = []

    # Search inside each root dir
    for root in source_roots:
        base_dir = Path(root)
        if not base_dir.exists():
            continue
        for pattern in include_patterns:
            all_files.extend(base_dir.rglob(pattern))

    # Apply exclusions
    if exclude_patterns:

        def is_excluded(path: Path) -> bool:
            return any(fnmatch.fnmatch(str(path), pat) for pat in exclude_patterns)

        all_files = [f for f in all_files if not is_excluded(f)]

    project_files = [str(f.resolve()) for f in all_files if f.is_file()]

    # Validate structure
    raw_structure = await create_document_structure(
        project_files, theme, project_insights
    )
    structure = DocumentationStructure(**raw_structure)

    # Attach context for documentation creation
    await attach_context(project_files)
    errors = []
    wiki_docs: List[str] = []
    for idx, topic in enumerate(structure.topics):
        page_content, ignored = await create_technical_topic_from_files(
            topic.files, f"{theme}: {topic.title}"
        )
        errors.append(ignored)

        doc_name = root_path / ".alquimia" / f"{idx}-{topic.title.replace(' ', '_')}.md"
        with open(doc_name, "w", encoding="utf-8") as f:
            f.write(page_content)

        wiki_docs.append(str(doc_name))

    logger.debug(errors)
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

    total_files = len(files)
    max_topics = 9
    min_files_per_topic = max(1, total_files // max_topics)
    max_files_per_topic = max(1, total_files // 2)

    project_structure_instructions = f"""
        Imagine an ideal **wiki-style documentation structure** for the given project source code files.
        Focus on generating a **hierarchical tree of topics** that organizes the code logically.

        Context:
        - Documentation main theme: `{theme}`
        - Project description: `{project_insights}`

        Requirements:
        1. Respond **strictly in JSON format**, no extra characters or explanations outside the JSON.
        2. The output must be a top-level object with a key `"topics"` containing a list of topic nodes.
        3. Each topic node must have:
           - `"title"`: a short descriptive string summarizing the topic.
           - `"files"`: a list of relevant source code files use relative paths
        4. Topic grouping rules:
           - Maximum number of topics: {max_topics}
           - Each topic must reference **at least {min_files_per_topic} and no more than {max_files_per_topic} files**.
           - No single file should appear in more than 3 topics.
           - Ensure **all project files** are included in at least one topic.
           - Consider file sizes, dependencies, and logical relevance when grouping.
           - Group topics to highlight:
               - Architecture
               - Modules and services
               - Integrations and APIs
               - Utilities, helpers, or common libraries
        5. JSON output must be **hierarchical if relevant**, showing subtopics or nested relationships based on logical grouping.

        Example JSON schema:
        {{
          "topics": [
            {{
              "title": "Architecture",
              "files": ["src/file1.py", "src/file2.py", "src/file3.py"]
            }},
            {{
              "title": "Services",
              "files": ["src/service1.py", "src/service2.py"]
            }}
          ]
        }}

        Constraints:
        - Do not include explanations or notes outside the JSON.
        - Prioritize logical, maintainable grouping over alphabetical order.
    """
    logger.debug(project_structure_instructions)
    response = await _client.infer(
        project_structure_instructions, extra_data={
        "project_structure": map_files_with_metadata(files),
    })

    stream_id = response["stream_id"]
    content = await _client.stream(stream_id)
    logger.debug(f"Project structure response: {content}")
    adict = extract_brace_block(content)
    logger.debug(adict)
    try:
        return json.loads(adict)
    except:
        logger.warning("Parsing failed! Trying replacing quotes..")
        return json.loads(adict.replace("'", '"'))


async def create_technical_topic_from_files(files: list[str], topic: str) -> str:
    """
    Create technical documentation based on the given set of relevant source code files and topic.
    """

    chunks = []
    ignored = []
    for f in files:
        file_path = Path(PROJECT_ROOT) / f.strip()
        try:
            with open(file_path, "r", encoding="utf-8") as _f:
                chunks.extend([_f.read()])
        except Exception as ex:
            logger.warning(f"Couldn't read file {file_path} - ignoring")
            ignored.append(file_path)
            continue

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

    return content, ignored


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
