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

with open('model/quic/EMNINST.pickle', 'rb') as fp:
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
        self.fl_model = keras_model
        self.model_weights = self.fl_model.load_weights(
            model_weights_path) if model_weights_path is not None else self.fl_model.get_weights()
        self.performance_hystory = {
            'performance_server': [],
            'performance_clients': []
        }
        self.train_flag = False
        self.server_task = self._loop.create_task(self._create_server_task())

    async def _create_server_task(self):
        '''Create running task for training
        '''
        while True:
            await asyncio.sleep(10)
            await self.train_model(self.find_clients())

    def find_clients(self):
        client_protocols = set()
        for cid, proto in self._protocols.items():
            if proto not in client_protocols:
                client_protocols.add(proto)
        return client_protocols

    async def train_model(self, client_protocols, num_rounds=10, min_clients=3):
        '''Function called by the server to train the model
        '''
        print(client_protocols)
        if len(client_protocols) >= min_clients:
            # Sample min_clients
            train_clients = random.sample(client_protocols, min_clients)
            print(train_clients)
            # Get remaining clients
            other_clients = client_protocols.difference(train_clients)
            print(other_clients)
            self.fl_model.set_weights(self.model_weights)
            # evaluate model
            eval_results = self.fl_model.evaluate(X_TEST, Y_TEST)
            # append to history
            self.performance_hystory['performance_server'].append(eval_results)
            # Train for number of rounds
            for rnd in range(num_rounds):
                if client_protocols.issubset(self.find_clients()):
                    # Get the tasks for client
                    fit_tasks = await self.fit_round(train_clients, rnd)
                    sample_size = sum(task.result()[1]
                                      for task in fit_tasks if task.result())
                    new_weights = []
                    # Add the weighted weights from each client
                    for task in fit_tasks:
                        results = task.result()
                        if results:
                            weighted = []
                            for layer in results[0]:
                                weighted.append((results[1]/sample_size)*layer)
                            if not new_weights:
                                new_weights = weighted
                            else:
                                for index in range(len(new_weights)):
                                    new_weights[index] += weighted[index]
                    if new_weights:
                        self.model_weights = new_weights
                else:
                    logging.info(
                        'Finished on round %s because not all clients are available', str(rnd))
                    break
            # self.finish_training(train_clients)

            if other_clients:
                wait_task.cancel()
                try:
                    await wait_task
                except asyncio.CancelledError:
                    logging.info("wait_task is cancelled now")

    def finish_training(self, clients):
        for client in clients:
            stream_id = min(client._quic._streams.keys())
            reader, writer = client._stream_transport.get(stream_id)
            writer.write_eof()

    async def fit_round(self, clients, rnd):
        logging.info('Training round %s', str(rnd))
        tasks = []
        # Create tasks for each client
        for client in clients:
            tasks.append(self._loop.create_task(self.train(client)))
        # wait for tasks to finish
        for task in tasks:
            await task
        return tasks

    async def train(self, client):
        '''One round of training for a client
        '''
        try:
            # End the wait task
            client._wait_task.cancel()
            try:
                await client._wait_task
            except asyncio.CancelledError:
                logging.info("_wait_task is cancelled for %s", client)
            # get stream id
            stream_id = min(client._quic._streams.keys())
            # get the reader and writer
            client.write(b'Sending Weights\n', stream_id)

            weight_streams = []
            for i in range(len(self.model_weights)):
                weight_streams[i] = client.create_stream()

            ans = await send_model_weights(self.model_weights, reader, writer)
            if ans:
                while True:
                    # wait for training to finish
                    line = await asyncio.wait_for(reader.readline(), timeout=60)
                    logging.info(line)
                    if line == b'Finished Round\n':
                        writer.write(b'Send new weights\n')
                        break
                new_model_weights = await get_model_weights(reader, writer)
                # get sample size
                line = await asyncio.wait_for(reader.readline(), timeout=60)
                sample_size = int(line)
                return (new_model_weights, sample_size)
        except asyncio.TimeoutError:
            logging.warning('One client did not respond in time')

    async def say_wait(self, clients):
        for client in clients:
            stream_id = min(client._quic._streams.keys())
            reader, writer = client._stream_transport.get(stream_id)
            writer.write(b'Wait for instructions\n')
        await asyncio.sleep(10)

    async def signal_waiting(self, clients):
        while True:
            await self.say_wait(clients)


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_transport = dict()
        self._ack_waiter: Optional[asyncio.Future[None]] = None
        self._wait_task = None

    def quic_event_received(self, event: QuicEvent) -> None:
        """
        Called when a QUIC event is received.
        Reimplement this in your subclass to handle the events.
        """
        if isinstance(event, ConnectionTerminated):
            print(event)
            for reader in self._stream_readers.values():
                reader.feed_eof()
        elif isinstance(event, StreamDataReceived):
            if self._ack_waiter is not None:
                print(event)
                waiter = self._ack_waiter
                self._ack_waiter = None
                waiter.set_result(None)

        elif isinstance(event, HandshakeCompleted):
            logging.info(event)
            self.initiate_comunication()

    def initiate_comunication(self):
        stream_id = self.create_stream()
        self._wait_task = self._loop.create_task(self.say_wait(stream_id))

    async def say_wait(self, stream_id):
        while True:
            self.write(b'Wait for instructions\n', stream_id)
            await asyncio.sleep(10)

    def create_stream(self):
        stream_id = self._quic.get_next_available_stream_id()
        return stream_id

    async def write(self, data, stream_id, end_stream=False):
        self._quic.send_stream_data(stream_id, data, end_stream)
        self.transmit()
        waiter = self._loop.create_future()
        self._ack_waiter = waiter
        self.transmit()
        return await asyncio.shield(waiter)


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
