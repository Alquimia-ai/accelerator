# main.py 

import os
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional

import markdown
from io import BytesIO
from weasyprint import HTML, CSS
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError

from fastmcp import FastMCP, Client, Context
from loguru import logger

from client import AlquimiaClient

# --- Set up ---

mcp = FastMCP("Report generator MCP for VO")

load_dotenv()

ALQUIMIA_API_KEY = os.environ.get("ALQUIMIA_API_KEY")
ALQUIMIA_URL = os.environ.get("ALQUIMIA_URL")
AGENT_ID = os.environ.get("AGENT_ID", "vo_report_generator")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "chat")

VO_MCP_URL = os.environ.get("VO_MCP_URL") # "http://0.0.0.0:8000/mcp"

BUCKET_NAME = os.environ.get("BUCKET_NAME", "vo-reports")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")


# --- Alquimia client ---

SESSION_ID = uuid.uuid4()

alquimia_client = AlquimiaClient(
    base_url=ALQUIMIA_URL,
    session_id=str(SESSION_ID),
    assistant_id=AGENT_ID,
    source=CHANNEL_ID,
    api_key=ALQUIMIA_API_KEY,
)


# --- VO MCP client ---

vo_client = Client(VO_MCP_URL)

async def query_shapefiles(polygon_coordinates, buffer_km):
    shapefile_data = []
    async with vo_client:
        try:
            result = await vo_client.call_tool("calculate_polygon_area", {"polygon_coordinates": polygon_coordinates})
            _ = f"The polygon under study has a total area of {result.data} km².\n---\n"
            shapefile_data.append(_)

            result = await vo_client.call_tool("get_red_list_in_polygon", {"polygon_coordinates": polygon_coordinates, "data_source": "IUCN"})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("report_key_area_overlap", {"polygon_coordinates": polygon_coordinates, "buffer_km": buffer_km})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("report_habitat_overlap", {"polygon_coordinates": polygon_coordinates})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("report_exploitation_area_overlap", {"polygon_coordinates": polygon_coordinates})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("find_nearby_coastal_communities", {"polygon_coordinates": polygon_coordinates, "buffer_km": buffer_km})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("get_human_activity_in_polygon", {"polygon_coordinates": polygon_coordinates})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("calculate_social_lag_in_polygon", {"polygon_coordinates": polygon_coordinates})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("get_fauna_in_polygon", {"polygon_coordinates": polygon_coordinates})
            shapefile_data.append(result.data+"-"*3+"\n")

            result = await vo_client.call_tool("estimate_blue_carbon_mangroves", {"polygon_coordinates": polygon_coordinates})

            _ = (
                "Estimates of blue carbon metrics for mangroves within a given polygon:"
                f"- Total area covered by mangroves in hectares: {result.data["mangrove_area_ha"]}"
                f"- Total carbon stock (Mg C): {result.data["total_carbon_Mg"]}"
                f"- Annual carbon sequestration (Mg C/year): {result.data["annual_sequestration_Mg"]}"
                f"- Estimated uncertainty of carbon stock (± Mg C): {result.data["total_carbon_error"]}"
                f"- CO₂ equivalent in metric tons (tCO₂e): {result.data["CO2e_t"]}"
                "---"
                )
            shapefile_data.append(_)

        except Exception as e:
            logger.exception(f"❌ Unexpected error running VO MCP")
            raise

        logger.debug(f"✅ Successfully gathered shapefile data")
        return "\n".join(shapefile_data)


# --- PDF conversion ---

def markdown_to_pdf(md_content: str) -> bytes:
    """
    Convert Markdown content to PDF bytes.
    
    Parameters
    ----------
    md_content : str
        Markdown content to convert
        
    Returns
    -------
    bytes
        PDF content as bytes
    """
    try:
        # Convert Markdown to HTML
        html_content = markdown.markdown(
            md_content, 
            extensions=['tables', 'fenced_code', 'toc', 'codehilite']
        )
        
        # Add basic CSS styling for better PDF appearance
        css_style = """
        @page {
            margin: 2cm;
            size: A4;
        }
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 1em;
            border-radius: 5px;
            overflow-x: auto;
        }
        blockquote {
            border-left: 4px solid #3498db;
            margin: 1em 0;
            padding-left: 1em;
            color: #666;
        }
        """
        
        # Create full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Vital Oceans Report</title>
            <style>{css_style}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Convert HTML to PDF
        pdf_buffer = BytesIO()
        HTML(string=full_html).write_pdf(pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        logger.debug("✅ Successfully converted Markdown to PDF")
        return pdf_bytes
        
    except Exception as e:
        logger.exception(f"❌ Error converting Markdown to PDF: {e}")
        raise


# --- S3 connection ---

def upload_pdf_to_s3(
    pdf_content: bytes,
    bucket_name: str,
    endpoint_url: str,
    file_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: str = 'us-east-1',
    make_presigned_url: bool = True,
    presign_expires: int = 86400  # 24 hours
) -> str:
    """
    Upload PDF content to S3-compatible storage (e.g., MinIO) and return a URL.
    
    Parameters
    ----------
    pdf_content : bytes
        PDF content as bytes
    bucket_name : str
        Name of the S3 bucket
    endpoint_url : str
        S3 endpoint URL
    file_name : Optional[str]
        Name for the file. If None, generates timestamp-based name
    aws_access_key_id : Optional[str]
        AWS access key ID
    aws_secret_access_key : Optional[str]
        AWS secret access key
    region_name : str
        AWS region name
    make_presigned_url : bool
        Whether to generate presigned URL
    presign_expires : int
        Presigned URL expiration time in seconds
        
    Returns
    -------
    str
        URL to access the uploaded PDF
    """
    try:
        # Create S3 client with path-style and v4 signatures (works best with MinIO)
        session = boto3.session.Session()
        s3_client = session.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=Config(
                s3={'addressing_style': 'path'},
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'standard'},
            ),
        )

        # File naming
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"report_{timestamp}.pdf"
        if not file_name.endswith('.pdf'):
            file_name += '.pdf'

        # Put object
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=pdf_content,
            ContentType='application/pdf',
        )

        # URL to return
        if make_presigned_url:
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': file_name},
                ExpiresIn=presign_expires,
            )
        else:
            # Path-style public URL (works if bucket policy allows public read)
            base = endpoint_url.rstrip('/')
            url = f"{base}/{bucket_name}/{file_name}"

        logger.debug(f"✅ Successfully uploaded {file_name} to S3")
        return url

    except NoCredentialsError:
        logger.error("❌ Error: AWS credentials not found")
        raise
    except ClientError:
        logger.exception("❌ AWS/MinIO ClientError occurred")
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        raise


# --- Write report tool ---

@mcp.tool
async def mpa_feasibility_report(
    polygon_coordinates: List[dict],
    ctx: Context,
    buffer_km: float = 10.0,
    language: str = "spanish",
    custom_requirements: Optional[str] = None
) -> str:
    """
    Generates a structured report assessing the feasibility of designating the proposed polygonal area as a Marine Protected Area.
    The report is generated as a PDF file, and the resulting PDF URL is sent to the client UI via the tool context (not returned).

    Parameters
    ----------
    polygon_coordinates : List[dict]
        Coordinate pairs representing the vertices of the polygon.
        Format: [{"lat": float, "lng": float}, ...]
    buffer_km : float, optional
        Buffer distance (in kilometers) to expand the polygon before analysis. Default is 10.
    language : str, optional
        Language in which the report should be written. Default is "spanish".
    custom_requirements : str, optional
        Specific requirements, constraints, or guidance to tailor the content of the report. Default is None.

    Returns
    -------
    str
        A confirmation message.
    """

    # Get shapefile data from VO MCP
    shapefile_data = await query_shapefiles(
        polygon_coordinates=polygon_coordinates,
        buffer_km=buffer_km,
    )

    # Build user query
    if language.lower() == "spanish":
        query = (
            "Escribe un informe en español que evalúe la factibilidad de "
            "designar el área bajo estudio como un Área Marina Protegida."
        )
    else:
        query = (
            f"Write a report in {language.title()} assessing the feasibility of "
            "designating the study area as a Marine Protected Area."
        )

    if custom_requirements:
        query += f"\n\n{custom_requirements}"

    # Report generation
    response = await alquimia_client.infer(
        query,
        extra_data={
            "polygon_coordinates": polygon_coordinates,
            "buffer_km": buffer_km,
            "language": language.title(),
            "shapefile_data": shapefile_data,
        },
    )
    stream_id = response["stream_id"]
    report_md = await alquimia_client.stream(stream_id)

    # Convert Markdown to PDF
    logger.info("Converting Markdown report to PDF...")
    pdf_content = markdown_to_pdf(report_md)

    # Upload PDF to S3 and get URL
    logger.info("Uploading PDF to S3...")
    url = upload_pdf_to_s3(
        pdf_content=pdf_content,
        bucket_name=BUCKET_NAME,
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name="nyc3",
    )

    logger.info("✅ Report URL successfully generated.")

    # Send URL via context
    if ctx:
        await ctx.info("MPA_REPORT_READY", extra={"url": url})

    return "Report was generated successfully."


# --- Run MCP ---

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        log_level="info"
        )