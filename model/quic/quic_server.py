import sys
import argparse
import asyncio
import importlib
import logging
import time
from collections import deque
from email.utils import formatdate
from typing import Any, Callable, Deque, Dict, List, Optional, Union, cast, Text, Tuple
import tensorflow as tf
import wsproto
import wsproto.events
from quic_logger import QuicDirectoryLogger
import pickle
import aioquic
import numpy as np
from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.quic.connection import NetworkAddress, QuicConnection
from aioquic.h0.connection import H0_ALPN, H0Connection
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, H3Event, HeadersReceived
from aioquic.h3.exceptions import NoAvailablePushIDError
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import DatagramFrameReceived, ProtocolNegotiated, QuicEvent, StreamDataReceived, ConnectionTerminated, PingAcknowledged, HandshakeCompleted
from aioquic.tls import SessionTicket
from common import send_model_weights, get_model_weights

from fl_model import create_keras_model
try:
    import uvloop
except ImportError:
    uvloop = None

with open('EMNINST.pickle', 'rb') as fp:
    a = pickle.load(fp)
    for i in range(100):
        key = next(iter(a))
    X_TEST, Y_TEST = a[key]
    for i in range(400):
        key = next(iter(a))
        x, y = a[key]
        np.concatenate((X_TEST, x), axis=0)
        np.concatenate((Y_TEST, y), axis=0)


class FLServer:

    def __init__(self, keras_model: tf.keras.models.Model, model_weights_path: str = None, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self._loop = asyncio.get_event_loop()
        self.fl_model = keras_model
        self.model_weights = self.fl_model.load_weights(
            model_weights_path) if model_weights_path is not None else self.fl_model.get_weights()
        # create QUIC logger

        self.quic_logger = QuicDirectoryLogger(
            "C:\\Users\\Eugen\\OneDrive - King's College London\\thesis\\model\\quic\\logs")

        self.quic_configuration = QuicConfiguration(
            is_client=False,
            quic_logger=self.quic_logger,
            idle_timeout=185.0
        )

        # load SSL certificate and key
        self.quic_configuration.load_cert_chain("c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_cert.pem",
                                                "c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_key.pem")

        self.ticket_store = SessionTicketStore()

    def stream_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        print('Create stream')
        self._loop.create_task(self._stream_handler(reader, writer))

    async def _stream_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        status = False
        while True:
            if reader.at_eof():
                writer.write_eof()
                print('break async')
                break
            line = await reader.readline()
            print(line)
            # if line==b'Ready to work\n':
            #     while not status:
            #         time.sleep(10)
            #         writer.write(b'Wait your turn\n')

            if line == b'Ready for training\n':
                await send_model_weights(self.model_weights, reader, writer)
            elif line == b'Finished Training\n':
                writer.write(b'Send new weights\n')
                new_model_weights = await get_model_weights(reader, writer)
                self.model_weights = new_model_weights
                self.fl_model.set_weights(self.model_weights)
                print(self.fl_model.evaluate(test_images, test_labels))

            # writer.write(b'Hello from stream handler\n')
            # writer.write(b'')
            # if line == b'Last\n':
            #     print('write EOF')
            #     writer.write_eof()
            #     reader.feed_eof()

    def run(self):
        self._loop.run_until_complete(
            serve(
                'localhost',
                4433,
                configuration=self.quic_configuration,
                create_protocol=NewProtocol,
                session_ticket_fetcher=self.ticket_store.pop,
                session_ticket_handler=self.ticket_store.add,
                stream_handler=self.stream_handler
            )
        )
        try:
            self._loop.run_forever()
        except KeyboardInterrupt:
            pass


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
            reader = self._stream_readers.get(event.stream_id, None)
            if reader is None:
                reader, writer = self._create_stream(event.stream_id)
                self._stream_handler(reader, writer)
            reader.feed_data(event.data)
            if event.end_stream:
                reader.feed_eof()


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
    with tf.device('cpu:0'):
        server = FLServer(create_keras_model())
        server.run()
