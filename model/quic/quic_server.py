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
from pathlib import Path

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

with open('model/quic/EMNINST_TEST.pickle', 'rb') as fp:
    a = pickle.load(fp)
    X_TEST, Y_TEST = None, None
    for key in a:
        if X_TEST is None and Y_TEST is None:
            X_TEST, Y_TEST = a[key]
        else:
            x, y = a[key]
            np.concatenate((X_TEST, x), axis=0)
            np.concatenate((Y_TEST, y), axis=0)


class Server(QuicServer):

    def __init__(self, keras_model: tf.keras.models.Model, model_weights_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fl_model = keras_model
        path = 'model/quic/server_weights'
        f = Path(path)
        if f.is_file():
            print('asdasd')
            with open(path, 'rb') as fp:
                self.model_weights = pickle.load(fp)
        else:
            self.model_weights = self.fl_model.get_weights()
        path = 'model/quic/history.pickle'
        f = Path(path)
        if f.is_file():
            print('asdasd')
            with open(path, 'rb') as fp:
                self.performance_hystory = pickle.load(fp)
        else:
            self.performance_hystory = []
        self.train_flag = False
        self.server_task = self._loop.create_task(self._create_server_task())

    async def _create_server_task(self):
        '''Create running task for training
        '''
        i = 0
        while True:
            await asyncio.sleep(5)
            print(i)
            await self.train_model(self.find_clients())
            i += 1

    def find_clients(self):
        client_protocols = set()
        for cid, proto in self._protocols.items():
            if proto not in client_protocols:
                client_protocols.add(proto)
        return client_protocols

    async def train_model(self, client_protocols, num_rounds=10, min_clients=10):
        '''Function called by the server to train the model
        '''
        if len(self.performance_hystory) == 30:
            with open('model/quic/history.pickle', 'wb') as fp:
                pickle.dump(self.performance_hystory, fp)
            sys.exit()
        # print(client_protocols)
        if len(client_protocols) >= min_clients:
            history = {
                'client_info': [],
                'accuracy': None
            }
            # Sample min_clients
            train_clients = random.sample(client_protocols, min_clients)
            # PROB
            # print(train_clients)
            # Get remaining clients
            other_clients = client_protocols.difference(train_clients)
            # print(other_clients)
            # Create task to make others wait
            if other_clients:
                wait_task = self._loop.create_task(
                    self.signal_waiting(other_clients))

            if client_protocols.issubset(self.find_clients()):
                # Get the tasks for client
                fit_tasks = await self.fit_round(train_clients)
                # Prob
                terminated_tasks = []
                for task in fit_tasks:
                    if task.result() is not None:
                        weights = task.result()[0]
                        # with x% chance drop the weights
                        samples = task.result()[1]
                        start = task.result()[2]
                        end = task.result()[3]
                        history['client_info'].append({
                            # 'weights': weights,
                            # 'samples': samples,
                            'send_time': (start, end)
                        })
                        if task.result()[1] is not None:
                            terminated_tasks.append((weights, samples))
                    else:
                        history['client_info'].append(None)

                sample_size = sum(task[1]
                                  for task in terminated_tasks)
                new_weights = []
                # Add the weighted weights from each client
                print(len(terminated_tasks))
                for task in terminated_tasks:
                    weighted = []
                    for layer in task[0]:
                        weighted.append((task[1]/sample_size)*layer)
                    if not new_weights:
                        new_weights = weighted
                    else:
                        for index in range(len(new_weights)):
                            new_weights[index] += weighted[index]
                if new_weights:
                    self.model_weights = new_weights
            self.fl_model.set_weights(self.model_weights)
            # evaluate model
            eval_results = self.fl_model.evaluate(X_TEST, Y_TEST)
            history['accuracy'] = eval_results
            print(eval_results)
            self.performance_hystory.append(history)
            if other_clients:
                wait_task.cancel()
                try:
                    await wait_task
                except asyncio.CancelledError:
                    logging.info("wait_task is cancelled now")
            self.finish_training(train_clients)
        else:
            # Wait for min clients
            await self.say_wait(client_protocols)

    def finish_training(self, clients):
        for client in clients:
            stream_id = min(client._quic._streams.keys())
            reader, writer = client._stream_transport.get(stream_id)
            writer.write_eof()
            client.close()

    async def fit_round(self, clients):
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
            # get stream id
            stream_id = min(client._quic._streams.keys())
            # get the reader and writer
            reader, writer = client._stream_transport.get(stream_id)
            writer.write(b'Proceed to Training\n')

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
                return new_model_weights, sample_size, ans[0], ans[1]
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

        quic_logger = QuicDirectoryLogger(
            "model\\quic\\logs")

        quic_configuration = QuicConfiguration(
            is_client=False,
            quic_logger=quic_logger
        )

        # load SSL certificate and key
        quic_configuration.load_cert_chain("model/quic/ssl_cert.pem",
                                           "model/quic/ssl_key.pem")

        ticket_store = SessionTicketStore()
        _loop = asyncio.get_event_loop()
        _loop.run_until_complete(
            serve_new(
                '0.0.0.0',
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
