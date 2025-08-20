from fastmcp import FastMCP
from fastmcp.server.context import Context
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from loguru import logger
from dataclasses import dataclass
from dependencies.tickets import get_tickets_provider
from models.ticket import Ticket

# Initialize FastMCP server
mcp = FastMCP(
    "arsat-customer-care",
    dependencies=[
        "httpx",
        "pydantic",
    ],
)


@dataclass
class MaximoContext:
    pluspcustomer: str


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint for the MCP server."""
    return PlainTextResponse("OK")


@mcp.tool
async def get_active_tickets(context: Context):
    """
    Get active tickets for the current organization

    Returns
    -------
    list
        A list of open tickets
    """

    customer_id = await validate_session_and_get_customer_id(context)

    if not customer_id:
        logger.error("Not found customer_id")
        raise ValueError("Customer ID not found in context")

    logger.debug(f"Get active tickets was called! - pluspcustomer: {customer_id}")
    tickets_provider = get_tickets_provider()()
    tickets = await tickets_provider.get_active_tickets(customer_id)
    return [t.model_dump() for t in tickets]


@mcp.tool
async def get_ticket_by_id(ticket_id: str, context: Context):
    """
    Get a ticket by its ID. Returns {} if not found

    Parameters
    ----------
    ticket_id : str
        The ID of the ticket, e.g., SS-575460

    Returns
    -------
    dict
        The ticket associated with the ID. Returns an empty dict if not found
    """

    customer_id = await validate_session_and_get_customer_id(context)

    logger.debug(
        f"Get ticket by id was called! - ticket_id: {ticket_id} - pluspcustomer: {customer_id}"
    )

    tickets_provider = get_tickets_provider()()
    ticket = await tickets_provider.get_ticket_by_id(ticket_id, customer_id)
    if ticket:
        return ticket.model_dump()
    return {}


@mcp.tool
async def get_classification(query: str, Context: Context):
    """
    Runs a clasification query agains a text classification model.
    Parameters
    ----------
    query : str
        The text to classify.

    Returns
    -------
    dict
        The classification result, e.g., {"category": "technical", "confidence": 0.95}
    """
    ## TODO: Implement with alquimia core


@mcp.tool
async def create_ticket(context: Context, ticket : Ticket):
    """
    Create a new ticket in the system.

    Parameters
    ----------
    ticket : Ticket
        The ticket data to create.

    Returns
    -------
    Ticket
        The created ticket object.
    """
    
    await validate_session_and_get_customer_id(context)
    
    tickets_provider = get_tickets_provider()()
    created_ticket = await tickets_provider.create_ticket(ticket)
    if created_ticket:
        return created_ticket
    return None


async def validate_session_and_get_customer_id(context: Context) -> str:
    """
    Valida la sesión del usuario y retorna el customer_id.

    Raises:
        ValueError: Si la sesión no es proporcionada, es cancelada o si el customer_id no se encuentra.
    """
    result = await context.elicit(
        message="Provide your active session", response_type=MaximoContext
    )

    if result.action == "decline":
        await context.error("Active session not provided")
        raise ValueError("Active session not provided")

    if result.action == "cancel":
        await context.error("Active session request was cancelled")
        raise ValueError("Active session request was cancelled")
    
    customer_id = result.data.pluspcustomer
    if not customer_id:
        logger.error("Not found customer_id")
        raise ValueError("Customer ID not found in context")
        
    return customer_id


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/consulting",
        log_level="info",
    )
