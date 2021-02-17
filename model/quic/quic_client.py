import argparse
import asyncio
import logging
import os
import pickle
import ssl
import time
from collections import deque
from typing import BinaryIO, Callable, Deque, Dict, List, Optional, Union, cast, Tuple
from urllib.parse import urlparse

import wsproto
import wsproto.events
from quic_logger import QuicDirectoryLogger

import aioquic
from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.h0.connection import H0_ALPN, H0Connection
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import (
    DataReceived,
    H3Event,
    HeadersReceived,
    PushPromiseReceived,
)
from aioquic.quic.configuration import QuicConfiguration
from aioquic.tls import CipherSuite, SessionTicket
from aioquic.quic.events import DatagramFrameReceived, ProtocolNegotiated, QuicEvent, StreamDataReceived, ConnectionTerminated, PingAcknowledged, HandshakeCompleted, ConnectionIdIssued

try:
    import uvloop
except ImportError:
    uvloop = None


HttpConnection = Union[H0Connection, H3Connection]

USER_AGENT = "aioquic/" + aioquic.__version__


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # def transmit(self) -> None:
    #     """
    #     Send pending datagrams to the peer and arm the timer if needed.
    #     """
    #     self._transmit_task = None

    #     # send datagrams
    #     for data, addr in self._quic.datagrams_to_send(now=self._loop.time()):
    #         self._transport.sendto(data, addr)

    #     # re-arm timer
    #     timer_at = self._quic.get_timer()
    #     if self._timer is not None and self._timer_at != timer_at:
    #         self._timer.cancel()
    #         self._timer = None
    #     if self._timer is None and timer_at is not None:
    #         self._timer = self._loop.call_at(timer_at, self._handle_timer)
    #     self._timer_at = timer_at

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
            reader = self._stream_readers.get(event.stream_id, None)
            if reader is None:
                print('----------')
                reader, writer = self._create_stream(event.stream_id)
                self._stream_handler(reader, writer)
            reader.feed_data(event.data)
            if event.end_stream:
                reader.feed_eof()

        # elif isinstance(event, HandshakeCompleted):
        #     print(event)
        #     reader, writer = self.create_stream()
        #     writer.write(b'Hello server')
        #     # self._quic.send_stream_data(stream_id, b'Hello server', False)


async def run(configuration: QuicConfiguration, host: str, port: int) -> None:
    async with connect(
        host,
        port,
        configuration=configuration,
        create_protocol=NewProtocol
    ) as client:
        # await client.wait_connected()
        # await client.ping()
        reader, writer = await client.create_stream()
        print('----------')
        print(client._transmit_task)
        writer.write(b'Hello server')
        for i in range(5):
            writer.write(b'Hello\n')
            # line = await reader.readline()
            # print(line)
            # client.transmit()
            # print('Client sent Hello wait response')
            # line = await reader.readline()
            # print('client response received')
            # print(line)
            # time.sleep(1)
        # client.transmit()

        print('Client sent Last wait response')
        writer.write(b'Last\n')
        print('sent')
        #
        writer.write_eof()
        print('sent eof')
        print((await reader.read()).decode())
        # print('client response received')
        # line = await reader.readline()
        # print(line)
        # line = '1'
        # while line:
        #     print(1)
        #     line = await reader.readline()
        #     print(line)
        client.close()
        await client.wait_closed()
        print('closed conn')

if __name__ == "__main__":
    defaults = QuicConfiguration(is_client=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO,
    )

    # prepare configuration
    configuration = QuicConfiguration(
        is_client=True
    )

    configuration.verify_mode = ssl.CERT_NONE

    configuration.quic_logger = QuicDirectoryLogger(
        "C:\\Users\\Eugen\\OneDrive - King's College London\\thesis\\model\\quic\\logs")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        run(
            configuration=configuration,
            host='localhost',
            port=4433
        )
    )
