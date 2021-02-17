
import argparse
import asyncio
import importlib
import logging
import time
from collections import deque
from email.utils import formatdate
from typing import Any, Callable, Deque, Dict, List, Optional, Union, cast, Text, Tuple

import wsproto
import wsproto.events
from quic_logger import QuicDirectoryLogger

import aioquic
from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.quic.connection import NetworkAddress, QuicConnection
from aioquic.h0.connection import H0_ALPN, H0Connection
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, H3Event, HeadersReceived
from aioquic.h3.exceptions import NoAvailablePushIDError
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import DatagramFrameReceived, ProtocolNegotiated, QuicEvent, StreamDataReceived, ConnectionTerminated, PingAcknowledged, HandshakeCompleted
from aioquic.tls import SessionTicket

try:
    import uvloop
except ImportError:
    uvloop = None

SERVER_NAME = "aioquic/" + aioquic.__version__


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    #     if self._stream_handler is not None:
    #         self._stream_handler = self._stream_handler
    #     else:
    #         self._stream_handler = self.new_stream_handler

    # def new_stream_handler(self, reader, writer):
    #     print('New stream handler')
    #     currentstreamtask = self._loop.create_task(
    #         self._currentstreamhandler(reader, writer))

    # async def _currentstreamhandler(self, reader, writer):
    #     line = await reader.readline()
    #     print(line)

    def quic_event_received(self, event: QuicEvent) -> None:
        """
        Called when a QUIC event is received.
        Reimplement this in your subclass to handle the events.
        """
        # FIXME: move this to a subclass
        if isinstance(event, ConnectionTerminated):
            for reader in self._stream_readers.values():
                reader.feed_eof()
        elif isinstance(event, StreamDataReceived):
            print('---------------------')
            reader = self._stream_readers.get(event.stream_id, None)
            if reader is None:
                print('reader none')
                reader, writer = self._create_stream(event.stream_id)
                self._stream_handler(reader, writer)
            reader.feed_data(event.data)
            print(event.data, event)
            # if event.data == b'Last':
            #     print('server: send Hello Last')
            #     self._quic.send_stream_data(
            #         event.stream_id, b' Last Hello\n', False)
            #     print('server:sent')
            # else:
            #     print('server: send Hello')
            #     self._quic.send_stream_data(event.stream_id, b'Hello\n', False)
            #     print('server:sent')
            if event.end_stream:
                print(event)
                reader.feed_eof()

        elif isinstance(event, DatagramFrameReceived):
            print(event)

        elif isinstance(event, PingAcknowledged):
            print(event)

        elif isinstance(event, HandshakeCompleted):
            print(event)


def stream_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    print('Create stream')
    loop = asyncio.get_event_loop()
    loop.create_task(funct(reader, writer))


async def funct(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    # while True:
    #     if reader.at_eof():
    #         line = await reader.read()
    #         print(line)

    while True:
        print('in async')
        if reader.at_eof():
            writer.write_eof()
            print('break async')
            break
        line = await reader.readline()
        print(line)
        # writer.write(b'Hello from stream handler\n')
        # writer.write(b'')
        # if line == b'Last\n':
        #     print('write EOF')
        #     writer.write_eof()
        #     reader.feed_eof()


class SessionTicketStore:
    """
    Simple in-memory store for session tickets.
    """

    def __init__(self) -> None:
        self.tickets: Dict[bytes, SessionTicket] = {}

    def add(self, ticket: SessionTicket) -> None:
        self.tickets[ticket.ticket] = ticket

    def pop(self, label: bytes) -> Optional[SessionTicket]:
        return self.tickets.pop(label, None)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO,
    )

    # create QUIC logger

    quic_logger = QuicDirectoryLogger(
        "C:\\Users\\Eugen\\OneDrive - King's College London\\thesis\\model\\quic\\logs")

    configuration = QuicConfiguration(
        is_client=False,
        quic_logger=quic_logger
    )

    # load SSL certificate and key
    configuration.load_cert_chain("c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_cert.pem",
                                  "c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_key.pem")

    ticket_store = SessionTicketStore()

    if uvloop is not None:
        uvloop.install()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        serve(
            'localhost',
            4433,
            configuration=configuration,
            create_protocol=NewProtocol,
            session_ticket_fetcher=ticket_store.pop,
            session_ticket_handler=ticket_store.add,
            stream_handler=stream_handler
        )
    )
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
