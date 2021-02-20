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
import socket

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
from fl_model import create_keras_model
from common import send_model_weights, get_model_weights
try:
    import uvloop
except ImportError:
    uvloop = None
with open('EMNINST.pickle', 'rb') as fp:
    DATA = pickle.load(fp)
    KEYS = iter(DATA)


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def quic_event_received(self, event: QuicEvent) -> None:
        """
        Called when a QUIC event is received.
        Reimplement this in your subclass to handle the events.
        """
        # FIXME: move this to a subclass
        if isinstance(event, ConnectionTerminated):
            print(event)
            for reader in self._stream_readers.values():
                reader.feed_eof()
        elif isinstance(event, StreamDataReceived):
            print(len(event.data))
            reader = self._stream_readers.get(event.stream_id, None)
            if reader is None:
                reader, writer = self._create_stream(event.stream_id)
                self._stream_handler(reader, writer)
            reader.feed_data(event.data)
            if event.end_stream:
                reader.feed_eof()


def save_session_ticket(ticket: SessionTicket) -> None:
    """
    Callback which is invoked by the TLS engine when a new session ticket
    is received.
    """
    logging.info("New session ticket received")
    if session_ticket_path:
        with open(session_ticket_path, "wb") as fp:
            pickle.dump(ticket, fp)


async def train_one_round(model, reader, writer):
    writer.write(b'Ready for training\n')
    model_weights = await get_model_weights(reader, writer)
    model.set_weights(model_weights)
    # Load data
    x_train, y_train = DATA[next(KEYS)]
    # Start training
    writer.write(b'Start model train\n')
    model.fit(x_train, y_train, epochs=15, batch_size=40, validation_split=0.2)
    writer.write(b'Finished Training\n')
    # Wait for server to ack
    line = await reader.readline()
    if line == b'Send new weights\n':
        return await send_model_weights(model.get_weights(), reader, writer)


async def run(configuration: QuicConfiguration, host: str, port: int) -> None:
    async with connect(
        host,
        port,
        configuration=configuration,
        # create_protocol=NewProtocol,
        session_ticket_handler=save_session_ticket,
    ) as client:
        reader, writer = await client.create_stream()
        model = create_keras_model()
        # writer.write(b'Ready to work\n')
        # while True:
        #     line = await reader.readline()
        #     if line == b'Proceed to Training\n':
        #         break
        i = 20
        while i:
            tmp = await train_one_round(model, reader, writer)
            if tmp:
                i -= 1
        writer.write_eof()
        print('sent eof')
        print((await reader.read()))
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
        is_client=True,
        idle_timeout=185.0
    )

    configuration.verify_mode = ssl.CERT_NONE

    configuration.quic_logger = QuicDirectoryLogger(
        "C:\\Users\\Eugen\\OneDrive - King's College London\\thesis\\model\\quic\\logs")
    session_ticket_path = 'session_ticket.tick'
    try:
        with open(session_ticket_path, "rb") as fp:
            logging.info('Loading session ticket')
            configuration.session_ticket = pickle.load(fp)
    except FileNotFoundError:
        pass

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        run(
            configuration=configuration,
            host='localhost',
            port=4433
        )
    )
