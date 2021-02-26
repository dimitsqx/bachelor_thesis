import sys
import argparse
import asyncio
import importlib
import logging
import time
import random
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
from aioquic.asyncio.server import QuicServer
from aioquic.quic.connection import NetworkAddress, QuicConnection
from aioquic.h0.connection import H0_ALPN, H0Connection
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, H3Event, HeadersReceived
from aioquic.h3.exceptions import NoAvailablePushIDError
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import DatagramFrameReceived, ProtocolNegotiated, QuicEvent, StreamDataReceived, ConnectionTerminated, PingAcknowledged, HandshakeCompleted
from aioquic.tls import SessionTicket, SessionTicketFetcher, SessionTicketHandler
from aioquic.asyncio.protocol import QuicStreamHandler, QuicStreamAdapter
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


class Server(QuicServer):

    def __init__(self, keras_model: tf.keras.models.Model, model_weights_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('in init')
        self.fl_model = keras_model
        self.model_weights = self.fl_model.load_weights(
            model_weights_path) if model_weights_path is not None else self.fl_model.get_weights()
        self.performance_hystory = {
            'performance_server': [],
            'performance_clients': []
        }
        self.train_flag = False
        self.server_task = self._loop.create_task(self.train_model())

    async def train_model(self, num_rounds=10):
        print('in train model')
        min_clients = 1
        while True:
            print('in while')
            client_proto = set()
            for cid, proto in self._protocols.items():
                if proto not in client_proto:
                    client_proto.add(proto)
            print(client_proto)
            if len(client_proto) >= min_clients:
                print('-----------------')
                train_clients = random.sample(client_proto, min_clients)
                print(train_clients)
                other_clients = client_proto.difference(train_clients)
                print(other_clients)
                wait_task = self._loop.create_task(
                    self.say_wait(other_clients))
                print('----------------------')
                self.fl_model.set_weights(self.model_weights)
                a = self.fl_model.evaluate(X_TEST, Y_TEST)
                # print(a)
                self.performance_hystory['performance_server'].append(a)
                for rnd in range(num_rounds):
                    a = await self.fit_round(client_proto, rnd)
                    # print(a)
                wait_task.cancel()
            else:
                for client in client_proto:
                    print(client._quic._streams.keys())
                    stream_id = min(client._quic._streams.keys())
                    reader, writer = client._stream_transport.get(stream_id)
                    writer.write(b'Wait for instructions\n')
                await asyncio.sleep(10)

    async def fit_round(self, clients, rnd):
        print('Training round ', str(rnd))
        tasks = []
        for client in clients:
            tasks.append(self._loop.create_task(self.train(client)))
        for task in tasks:
            print(task)
            await task
        return tasks

    async def train(self, client):
        print('train task')
        stream_id = stream_id = min(client._quic._streams.keys())
        reader, writer = client._stream_transport.get(stream_id)
        writer.write(b'Proceed to Training\n')
        print('sending weights')
        ans = await send_model_weights(self.model_weights, reader, writer)
        print('got answer', ans)
        if ans:
            while True:
                # wait for training to finish
                line = await reader.readline()
                print(line)
                if line == b'Finished Round\n':
                    writer.write(b'Send new weights\n')
                    break
            new_model_weights = await get_model_weights(reader, writer)
            print('got model_weights')
            self.model_weights = new_model_weights
        else:
            pass

        return

    async def say_wait(self, clients):
        while True:
            print('Signaling other to wait')
            for client in clients:
                print(client._quic._streams.keys())
                stream_id = min(client._quic._streams.keys())
                # client._quic.send_stream_data(
                #     stream_id, b'Wait for instructions\n')
                reader, writer = client._stream_transport.get(stream_id)
                writer.write(b'Wait for instructions\n')
            await asyncio.sleep(10)


# class FLServer:

#     def __init__(self, keras_model: tf.keras.models.Model, model_weights_path: str = None, *args, **kwargs):
#         # super().__init__(*args, **kwargs)
#         self._loop = asyncio.get_event_loop()
#         self.fl_model = keras_model
#         self.model_weights = self.fl_model.load_weights(
#             model_weights_path) if model_weights_path is not None else self.fl_model.get_weights()
#         # create QUIC logger

#         self.quic_logger = QuicDirectoryLogger(
#             "C:\\Users\\Eugen\\OneDrive - King's College London\\thesis\\model\\quic\\logs")

#         self.quic_configuration = QuicConfiguration(
#             is_client=False,
#             quic_logger=self.quic_logger,
#             idle_timeout=185.0
#         )

#         # load SSL certificate and key
#         self.quic_configuration.load_cert_chain("c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_cert.pem",
#                                                 "c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_key.pem")

#         self.ticket_store = SessionTicketStore()

#     def stream_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
#         print('Create stream')
#         self._loop.create_task(self._stream_handler(reader, writer))

#     async def _stream_handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
#         status = False
#         while True:
#             if reader.at_eof():
#                 writer.write_eof()
#                 print('break async')
#                 break
#             line = await reader.readline()
#             print(line)
#             # if line==b'Ready to work\n':
#             #     while not status:
#             #         time.sleep(10)
#             #         writer.write(b'Wait your turn\n')

#             if line == b'Ready for training\n':
#                 await send_model_weights(self.model_weights, reader, writer)
#             elif line == b'Finished Training\n':
#                 writer.write(b'Send new weights\n')
#                 new_model_weights = await get_model_weights(reader, writer)
#                 self.model_weights = new_model_weights
#                 self.fl_model.set_weights(self.model_weights)
#                 print(self.fl_model.evaluate(X_TEST, Y_TEST))

#             # writer.write(b'Hello from stream handler\n')
#             # writer.write(b'')
#             # if line == b'Last\n':
#             #     print('write EOF')
#             #     writer.write_eof()
#             #     reader.feed_eof()

#     def run(self):
#         self._loop.run_until_complete(
#             serve(
#                 'localhost',
#                 4433,
#                 configuration=self.quic_configuration,
#                 create_protocol=NewProtocol,
#                 session_ticket_fetcher=self.ticket_store.pop,
#                 session_ticket_handler=self.ticket_store.add,
#                 stream_handler=self.stream_handler
#             )
#         )
#         try:
#             self._loop.run_forever()
#         except KeyboardInterrupt:
#             pass


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_transport = dict()

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
            reader, writer = self._stream_transport.get(
                event.stream_id, (None, None))
            if reader is None:
                reader, writer = self._create_stream(event.stream_id)
                self._stream_handler(reader, writer)
            else:
                reader.feed_data(event.data)
            if event.end_stream:
                reader.feed_eof()
        elif isinstance(event, HandshakeCompleted):
            print(event)

    def _create_stream(
        self, stream_id: int
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        adapter = QuicStreamAdapter(self, stream_id)
        reader = asyncio.StreamReader()
        writer = asyncio.StreamWriter(adapter, None, reader, self._loop)
        self._stream_readers[stream_id] = reader
        self._stream_transport[stream_id] = (reader, writer)
        return reader, writer


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


async def serve_new(host: str,
                    port: int,
                    *,
                    configuration: QuicConfiguration,
                    create_protocol: Callable = QuicConnectionProtocol,
                    session_ticket_fetcher: Optional[SessionTicketFetcher] = None,
                    session_ticket_handler: Optional[SessionTicketHandler] = None,
                    retry: bool = False,
                    stream_handler: QuicStreamHandler = None,
                    ) -> Server:
    loop = asyncio.get_event_loop()

    _, protocol = await loop.create_datagram_endpoint(
        lambda: Server(
            configuration=configuration,
            create_protocol=create_protocol,
            session_ticket_fetcher=session_ticket_fetcher,
            session_ticket_handler=session_ticket_handler,
            retry=retry,
            stream_handler=stream_handler,
            keras_model=create_keras_model()
        ),
        local_addr=(host, port),
    )
    return cast(Server, protocol)

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO,
    )
    with tf.device('cpu:0'):
        # server = FLServer(create_keras_model())
        # server.run()

        quic_logger = QuicDirectoryLogger(
            "C:\\Users\\Eugen\\OneDrive - King's College London\\thesis\\model\\quic\\logs")

        quic_configuration = QuicConfiguration(
            is_client=False,
            quic_logger=quic_logger
        )

        # load SSL certificate and key
        quic_configuration.load_cert_chain("c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_cert.pem",
                                           "c:/Users/Eugen/OneDrive - King's College London/thesis/model/quic/ssl_key.pem")

        ticket_store = SessionTicketStore()
        _loop = asyncio.get_event_loop()
        _loop.run_until_complete(
            serve_new(
                'localhost',
                4433,
                configuration=quic_configuration,
                create_protocol=NewProtocol,
                session_ticket_fetcher=ticket_store.pop,
                session_ticket_handler=ticket_store.add
            )
        )
        try:
            _loop.run_forever()
        except KeyboardInterrupt:
            pass
