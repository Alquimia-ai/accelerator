import hashlib
import mimetypes
import os
import subprocess
import uuid

from fastmcp import FastMCP

from client import AlquimiaClient

mcp = FastMCP("Technical content generator")

API_KEY = os.environ.get("API_KEY", None)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
AGENT_ID = os.environ.get("AGENT_ID", "code_reviewer")
MAX_INPUT_BATCH_SIZE = os.environ.get("MAX_INPUT_BATCH_SIZE", 5000)

session_id = uuid.uuid4()

client = AlquimiaClient(
    BASE_URL,
    str(session_id),
    AGENT_ID,
    "chat",
    api_key=API_KEY,
)


@mcp.tool
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


@mcp.tool
async def attach_context(files: list[str]) -> str:
    """
    Attach the given set of files into the context
    """
    attachments = []
    for path in filter(os.path.isfile, files):
        content_type, _ = mimetypes.guess_type(path)
        size = os.path.getsize(path)

        # Generate hash from path
        # hash_digest = hashlib.sha256(path.encode()).hexdigest()  # shorten to 12 chars
        # _, ext = os.path.splitext(path)
        # hashed_filename = f"{hash_digest}{ext}"

        attachments.append(
            {
                "content_type": content_type,
                "content_size": size,
                "filename": os.path.basename(path),
            }
        )

    response = await client.infer("", attachments=attachments)
    attachment_ids = response["attachments"]
    stream_id = response["stream_id"]

    for idx, path in enumerate(filter(os.path.isfile, files)):
        await client.upload_attachment(stream_id, session_id, attachment_ids[idx], path)


@mcp.tool
async def create_technical_topic_from_files(files: list[str], topic: str,
                                           indications: str) -> str:
    """
    Create technical documentation based on the given set of codebase files.
    Send the topic to create, like "Architecture", or "Installation guide" and
    indications like: "Use mermaid diagrams when possible"
    """
    CLAUSES = [
        "Generate a technical wiki topic page for the given topic and indications",
        f"Topic: {topic}. Indications: {indications}",
        "The codebase file and actual code is being shared",
        "If the current wiki page state is shared, rewrite as needed without losing the
        original meaning",
        "Do not share false or misguiding information"
    ]
    for f in files:
        current_page = ""
        with open(f, "r") as _f:
            content = _f.read()

            chunked_content = split_text_by_chars(content)

            response = await client.infer(
                "\n".join(CLAUSES + [f"File: {os.path.basename(f)}. Code: {chunked_content[0]}", f"Current page: {current_page}"])
            )
            release_notes = await client.stream(stream_id)
            if len(chunked_content) > 1:
                for idx, chunk in enumerate(chunked_content[1:]):
                    response = await client.infer(
                        "\n".join(CLAUSES + [f"File: {os.path.basename(f)}. Code: {chunk}", f"Current page: {current_page}"])
                    )
                    release_notes = await client.stream(response["stream_id"])

    return release_notes

@mcp.tool
async def create_release_notes_from_diff(diff: str) -> str:
    """
    Create a release notes based on a git diff output
    """
    if not chunked_diff:
       return

    CLAUSES = [
        "Generate a release notes based on the given git diff",
        "If the current release notes state is shared, rewrite as needed without losing the
        original meaning",
        "Do not share false or misguiding information"
    ]

    chunked_diff = split_text_by_chars(diff)
    total_chunks = len(chunked_diff)
    response = await client.infer(
        "\n".join(CLAUSES + [f"Git diff 1 of {total_chunks}: {chunked_diff[0]}"])
    )

    release_notes = await client.stream(stream_id)
    if len(chunked_diff) > 1:
        for idx, chunk in enumerate(chunked_diff[1:]):
            response = await client.infer(
                "\n".join(CLAUSES + [
                    f"Current release notes: {release_notes}",
                    f"nGit diff {idx} of {total_chunks}: {chunk}"
                ])
            )
            release_notes = await client.stream(response["stream_id"])

    return release_notes


def split_text_by_chars(text, max_chars=MAX_INPUT_BATCH_SIZE):
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


if __name__ == "__main__":
    mcp.run()
