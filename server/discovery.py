"""
server/discovery.py -- advertises this RiceGuard AI server on the local
network via mDNS/DNS-SD (zeroconf), so the Android client can find it
automatically instead of the user typing in the PC's IP address.

Service type: _riceguard._tcp.local.
TXT record: app, version, api, name (see RiceGuardAdvertiser.start()).

Bound to whatever non-loopback IPv4 address(es) this machine has -- never
127.0.0.1/localhost, which would be useless to a phone on the same Wi-Fi
(see server/security.py's LocalNetworkOnlyMiddleware for the matching
server-side check on the request path). The advertisement is unregistered
on shutdown (see server/main.py's lifespan) so it doesn't linger in other
devices' mDNS caches after this process exits.
"""

from __future__ import annotations

import logging
import socket

from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

logger = logging.getLogger("riceguard.discovery")

SERVICE_TYPE = "_riceguard._tcp.local."
INSTANCE_NAME = "RiceGuard Server"


def _routing_table_address() -> str | None:
    """Asks the OS routing table which local address it would use to reach
    the internet -- a UDP "connect" that never actually sends a packet, just
    picks an outbound interface. This is the address a phone on the same
    Wi-Fi can actually reach."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _local_ipv4_addresses() -> list[str]:
    """This machine's LAN-facing IPv4 address(es), most-likely-correct
    first. A dev machine commonly has several non-loopback adapters at once
    (WSL, Hyper-V, Docker, a VPN client) whose addresses
    socket.getaddrinfo(hostname) happily returns alongside the real Wi-Fi
    one, in no meaningful order -- advertising one of those instead would
    make the phone "find" a server it can never actually reach. The
    routing-table trick above reliably picks the real outbound-facing (i.e.
    Wi-Fi) address, so it always goes first; any other private addresses are
    appended only as extra fallback candidates, never in place of it."""
    addresses: list[str] = []
    seen: set[str] = set()

    primary = _routing_table_address()
    if primary and not primary.startswith("127."):
        addresses.append(primary)
        seen.add(primary)

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127.") and addr not in seen:
                addresses.append(addr)
                seen.add(addr)
    except OSError:
        pass

    return addresses


class RiceGuardAdvertiser:
    """Owns one AsyncZeroconf instance + one registered ServiceInfo for the
    lifetime of the server process. Uses zeroconf's asyncio API (not the
    plain synchronous Zeroconf/register_service) because start()/stop() run
    from inside FastAPI's async lifespan, which itself runs on uvicorn's
    event loop -- calling the synchronous API from there blocks that loop
    while zeroconf's internal engine waits on a cross-thread future, which
    reliably raised zeroconf.exceptions.EventLoopBlocked in testing; await-ing
    the async API from the same loop instead of blocking it avoids that.

    A failure to start (no LAN address found, zeroconf/socket error) is
    logged and swallowed rather than raised -- a server that can't advertise
    itself should still serve requests from clients that already know its IP
    (manual entry stays a valid fallback), it just won't be auto-discoverable.
    """

    def __init__(self, port: int, version: str = "1.0.0"):
        self._port = port
        self._version = version
        self._zeroconf: AsyncZeroconf | None = None
        self._service_info: ServiceInfo | None = None

    async def start(self) -> None:
        addresses = _local_ipv4_addresses()
        if not addresses:
            logger.warning("mDNS advertisement skipped: no non-loopback IPv4 address found.")
            return

        properties = {
            "app": "riceguard",
            "version": self._version,
            "api": "/health",
            "name": INSTANCE_NAME,
        }

        try:
            zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
            info = ServiceInfo(
                type_=SERVICE_TYPE,
                name=f"{INSTANCE_NAME}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(a) for a in addresses],
                port=self._port,
                properties=properties,
                server=f"{socket.gethostname()}.local.",
            )
            await zc.async_register_service(info)
        except Exception:
            logger.exception("mDNS advertisement failed to start; the app will need manual IP entry.")
            return

        self._zeroconf = zc
        self._service_info = info
        logger.info(
            "mDNS: advertising '%s' on %s at %s:%d",
            INSTANCE_NAME, SERVICE_TYPE, addresses[0], self._port,
        )

    async def stop(self) -> None:
        if self._zeroconf is None:
            return
        try:
            if self._service_info is not None:
                await self._zeroconf.async_unregister_service(self._service_info)
            await self._zeroconf.async_close()
            logger.info("mDNS: RiceGuard Server advertisement stopped.")
        except Exception:
            logger.exception("mDNS advertisement did not shut down cleanly.")
        finally:
            self._zeroconf = None
            self._service_info = None
