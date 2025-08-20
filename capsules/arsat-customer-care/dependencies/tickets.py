from abc import ABC, abstractmethod
import datetime
from typing import List, Optional
from models.ticket import Ticket
import httpx
from loguru import logger
import json
import urllib.parse
from config.manager import settings


class TicketsProvider(ABC):
    async def query(self, oslc_where_clause: str) -> List[Ticket]:
        """Query tickets using an OSLC clause."""
        ...

    @abstractmethod
    async def get_active_tickets(self, customer_id: str) -> List[Ticket]:
        """Get all active tickets for a given customer."""
        ...

    @abstractmethod
    async def get_ticket_by_id(
        self, ticket_id: str, customer_id: str
    ) -> Optional[Ticket]:
        """Retrieve a single ticket by its ID and customer ID."""
        ...
    
    @abstractmethod
    async def create_ticket(self, ticket: Ticket) -> Ticket:
        """Create a new ticket in the system."""
        ...


class MaximoTicketsProvider(TicketsProvider):
    maximo_person_obj = "A_TKWL_IA_RH"

    def __init__(
        self, base_url, user_id, passwd, request_timeout=10.0, verify_ssl=True
    ):
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.verify_ssl = verify_ssl
        ticket_fields = ",".join(
            f"spi:{fld.lower()}"
            for fld in Ticket.model_fields
            if fld.lower() != "worklogs"
        )
        self.query_args = [
            f"oslc.select={ticket_fields},spi:worklog",
            f"_lid={user_id}",
            f"_lpwd={passwd}",
        ]

    async def query(self, oslc_where_clause) -> List[Ticket]:
        """
        We take `oslc_where_clause` (a string) and combine it with the base query_args.
        """
        ## Hardcode the first 10 results for now in a descending order
        final_args = self.query_args + [
            oslc_where_clause,
            "oslc.orderBy=-spi:statusdate",
            "oslc.pageSize=10",
        ]
        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            maximo_url = (
                f'{self.base_url}/{self.maximo_person_obj}?{"&".join(final_args)}'
            )
            logger.debug(f"Attemping to query tickets with {maximo_url}")
            response = await client.get(maximo_url, timeout=self.request_timeout)
            response.raise_for_status()
            r = response.json()
            tickets_data = r.get("rdfs:member", [])
            logger.debug(f"Tickets found: {tickets_data}")
            return [Ticket.from_maximo_oslc(t) for t in tickets_data]

    async def get_active_tickets(self, customer_id) -> List[Ticket]:
        encoded_statuses = urllib.parse.quote(
            json.dumps(settings.MAXIMO_OPEN_TICKET_STATUSES)
        ).replace("%2C%20", "%2C")
        where_clause = f"oslc.where=spi:status in {encoded_statuses} and spi:pluspcustomer=%22{customer_id}%22"
        return await self.query(where_clause)

    async def get_ticket_by_id(self, ticket_id, customer_id) -> Optional[Ticket]:
        where_clause = f'oslc.where=spi:ticketid="{ticket_id}" and spi:pluspcustomer="{customer_id}"'
        tickets = await self.query(where_clause)
        return tickets[0] if tickets else None
    
    async def create_ticket(self, ticket: Ticket) -> Ticket:
        url = f"{self.base_url}/{self.maximo_person_obj}" # Se utiliza esta url para crear tickets??

        payload = {
            "spi:class": ticket.CLASS,
            "spi:description": ticket.DESCRIPTION,
            # Debe ser el mismo customer del context de la sesión?
            "spi:pluspcustomer": ticket.PLUSPCUSTOMER,
            "spi:status": ticket.STATUS,
            # Estos dos campos deben ser generados por el sistema?
            "spi:statusdate": ticket.STATUSDATE, 
            "spi:ticketid": ticket.TICKETID,
        }

        # Filtramos los valores nulos del payload
        payload = {k: v for k, v in payload.items() if v is not None}

        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            logger.debug(f"Posting new ticket to {url} with payload: {payload}")
            response = await client.post(
                url,
                params=self.query_args, # Incluyen los parámetros de autenticación
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            created_ticket_data = response.json()
            logger.debug(f"Created ticket response: {created_ticket_data}")
            return Ticket.from_maximo_oslc(created_ticket_data)


class MaximoFakeTicketsProvider(TicketsProvider):

    def __init__(self, tickets_path="mocks/maximo_tickets.json"):
        self.tickets = json.load(open(tickets_path))["QueryA_TKWL_IA_RHResponse"][
            "A_TKWL_IA_RHSet"
        ]["TICKET"]

    async def get_active_tickets(self, customer_id) -> List[Ticket]:
        return [
            Ticket._dump(t)
            for t in self.tickets
            if t["Attributes"]["STATUS"]["content"]
            in settings.MAXIMO_OPEN_TICKET_STATUSES
            and t["Attributes"]["PLUSPCUSTOMER"]["content"] == customer_id
        ]

    async def get_ticket_by_id(self, ticket_id, customer_id) -> Optional[Ticket]:
        tickets = [
            Ticket.from_maximo_dump(t)
            for t in self.tickets
            if t["Attributes"]["TICKETID"]["content"] == ticket_id
            and t["Attributes"]["PLUSPCUSTOMER"]["content"] == customer_id
        ]
        if tickets and len(tickets) > 0:
            return tickets[0]
        return None
    
    async def create_ticket(self, ticket: Ticket) -> Ticket:
        new_ticketid = ticket.TICKETID or f"Fake-{int(datetime.now().timestamp() * 1000)}"

        new_ticket = {
            "rowstamp": f"RS-Fake-{(int(datetime.now().timestamp() * 1000))}",
            "Attributes": {
                "CLASS": {"content": ticket.CLASS or "SR"},
                "DESCRIPTION": {"content": ticket.DESCRIPTION or ""},
                "PLUSPCUSTOMER": {"content": ticket.PLUSPCUSTOMER or ""},
                "STATUS": {"content": ticket.STATUS or "NEW"},
                "STATUSDATE": {"content": ticket.STATUSDATE or datetime.now().isoformat()},
                "TICKETID": {"content": new_ticketid},
                "TICKETUID": {"content": len(self.tickets) + 1000, "resourceid": True},
            },
            "RelatedMbos": {"WORKLOG": []},
        }

        if ticket.WORKLOGS:
            for idx, wl in enumerate(ticket.WORKLOGS, start=1):
                new_wl = {
                    "rowstamp": f"RS-Fake-{(int(datetime.now().timestamp() * 1000) + idx)}",
                    "Attributes": {
                        "CREATEBY": {"content": wl.CREATEBY or "FAKEUSER"},
                        "CREATEDATE": {"content": wl.CREATEDATE or datetime.now().isoformat()},
                        "DESCRIPTION": {"content": wl.DESCRIPTION or ""},
                        "LOGTYPE": {"content": wl.LOGTYPE or "CLIENTNOTE"},
                        "LOGDESCRIPTION": {"content": wl.LOGDESCRIPTION or ""},
                        "MODIFYBY": {"content": wl.MODIFYBY or ""},
                        "MODIFYDATE": {"content": wl.MODIFYDATE or datetime.now().isoformat()},
                        "SITEID": {"content": wl.SITEID or "FAKESITE"},
                        "WORKLOGID": {"content": len(new_ticket["RelatedMbos"]["WORKLOG"]) + 1, "resourceid": True},
                        "CLIENTVIEWABLE": {"content": True}, 
                    },
                }
                new_ticket["RelatedMbos"]["WORKLOG"].append(new_wl)

        self.tickets.append(new_ticket)

        with open(self.tickets_path, "w") as f:
            json.dump(self.root, f, indent=2)

        return Ticket.from_maximo_dump(new_ticket)


def get_tickets_provider():
    def _get_maximo_provider() -> TicketsProvider:
        if settings.MAXIMO_BASE_URL:
            logger.debug("Using MaximoTicketsProvider with real Maximo instance")
            return MaximoTicketsProvider(
                settings.MAXIMO_BASE_URL,
                settings.MAXIMO_USER_ID,
                settings.MAXIMO_PASSWD,
                request_timeout=settings.MAXIMO_REQUEST_TIMEOUT,
                verify_ssl=settings.MAXIMO_HTTP_VERIFY_SSL,
            )
        else:
            logger.debug("Using MaximoFakeTicketsProvider with mock data")
            return MaximoFakeTicketsProvider()

    return _get_maximo_provider
