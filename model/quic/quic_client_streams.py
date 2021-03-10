import argparse
import asyncio
import logging
import os
import pickle
import ssl
import time
import tensorflow as tf
from collections import deque
from typing import BinaryIO, Callable, Deque, Dict, List, Optional, Union, cast, Tuple
from urllib.parse import urlparse
import socket

import wsproto
import wsproto.events
from quic_logger import QuicDirectoryLogger

import aioquic
from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol, QuicStreamHandler, QuicStreamAdapter
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
with open('model/quic/EMNINST.pickle', 'rb') as fp:
    DATA = pickle.load(fp)
    KEYS = iter(DATA)


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ack_waiter: Optional[asyncio.Future[None]] = None
        self.client_handler = ClientHandler()
        self._wait_training = None

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

            print(event)
            self.handle_message(event.data, event.stream_id)
            waiter = self._ack_waiter
            self._ack_waiter = None
            if waiter:
                waiter.set_result(None)

    def handle_message(self, message, stream_id):
        if message == b'Wait for instructions\n':
            logging.info('Waiting')
            return
        if message == b'Sending Weights\n':
            logging.info('Waiting for weights')
            return

    def create_stream(self):
        stream_id = self._quic.get_next_available_stream_id()
        self._quic._get_or_create_stream_for_send(stream_id)
        return stream_id

    async def write(self, data, stream_id, end_stream=False):
        self._quic.send_stream_data(stream_id, data, end_stream)
        waiter = self._loop.create_future()
        self._ack_waiter = waiter
        self.transmit()
        return await asyncio.shield(waiter)


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
    model_weights = await get_model_weights(reader, writer)
    model.set_weights(model_weights)
    # Load data
    x_train, y_train = DATA[next(KEYS)]
    # Start training
    writer.write(b'Start model train\n')
    model.fit(x_train, y_train, epochs=15, batch_size=40, validation_split=0.2)
    writer.write(b'Finished Round\n')
    # Wait for server to ack
    line = await reader.readline()
    if line == b'Send new weights\n':
        sent_weights = await send_model_weights(model.get_weights(), reader, writer)
        print('send sample size')
        writer.write('{0}\n'.format(len(x_train)).encode())


async def run(configuration: QuicConfiguration, host: str, port: int) -> None:
    async with connect(
        host,
        port,
        configuration=configuration,
        create_protocol=NewProtocol,
        session_ticket_handler=save_session_ticket,
    ) as client:
        # stream_id = client.create_stream()

        # print(client._quic._streams)
        # await client.write(b'Hello', stream_id1)
        # while True:
        #     line= await reader.readline()
        #     if line==b'Prepare for training\n':
        #         break
        # model = create_keras_model()
        # writer.write(b'Ready to work\n')
        # # print(client._quic.host_cid)
        # # print(client._quic._peer_cid)

        # while not reader.at_eof():
        #     line = await reader.readline()
        #     print(line)
        #     if line == b'Proceed to Training\n':
        #         tmp = await train_one_round(model, reader, writer)
        # writer.write_eof()
        # print('sent eof')
        # print((await reader.read()))
        # client.close()
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
    session_ticket_path = 'session_ticket.tick'
    try:
        with open(session_ticket_path, "rb") as fp:
            logging.info('Loading session ticket')
            configuration.session_ticket = pickle.load(fp)
    except FileNotFoundError:
        pass
    with tf.device('cpu:0'):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            run(
                configuration=configuration,
                host='localhost',
                port=4433
            )
        )
