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


from fl_model import create_keras_model
try:
    import uvloop
except ImportError:
    uvloop = None

with open('model/quic/EMNINST_TEST.pickle', 'rb') as fp:
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

    async def train_model(self, client_protocols, num_rounds=2, min_clients=2):
        '''Function called by the server to train the model
        '''
        print('All clients')
        print(client_protocols)
        if len(client_protocols) >= min_clients:
            # Sample min_clients
            train_clients = set(random.sample(client_protocols, min_clients))
            print('Train clients')
            print(train_clients)
            # Get remaining clients
            other_clients = client_protocols.difference(train_clients)
            print('Other clients')
            print(other_clients)
            self.fl_model.set_weights(self.model_weights)
            # evaluate model
            eval_results = self.fl_model.evaluate(X_TEST, Y_TEST)
            # append to history
            self.performance_hystory['performance_server'].append(eval_results)
            print(eval_results)
            # Train for number of rounds
            for rnd in range(num_rounds):
                if train_clients.issubset(self.find_clients()):
                    # Get the tasks for client
                    fit_tasks = await self.fit_round(train_clients, rnd)
                    print(fit_tasks)
                    terminated_tasks = [task.result()
                                        for task in fit_tasks if task.result() is not None and task.result()[1] is not None]
                    print('Training round finished')
                    print(terminated_tasks)
                    for index in range(len(self.model_weights)):
                        # index of  task that returned weights for this layer
                        task_returned_weights_idx = [i for i, e in enumerate(
                            terminated_tasks) if e[0][index] is not None]
                        print(task_returned_weights_idx)
                        # calculate sample size for tasks that returned weights for this layer
                        sample_size = sum(terminated_tasks[i][1]
                                          for i in task_returned_weights_idx)
                        print(sample_size)
                        # get the average sum
                        layer = None
                        for idx in task_returned_weights_idx:
                            weights = terminated_tasks[idx][0][index]

                            samples = terminated_tasks[idx][1]
                            if layer is not None:
                                layer += (weights*samples)/sample_size
                            else:
                                layer = (weights*samples)/sample_size
                        if layer is not None:
                            self.model_weights[index] = layer
                else:
                    logging.info(
                        'Finished on round %s because not all clients are available', str(rnd))
                    break
            print('Finished all rounbds')
            self.finish_training(
                train_clients.intersection(self.find_clients()))

    def finish_training(self, clients):
        print(clients)
        for client in clients:
            client.create_wait_task()

    async def fit_round(self, clients, rnd):
        logging.info('Training round %s', str(rnd))
        tasks = []
        # Create tasks for each client
        for client in clients:
            tasks.append(self._loop.create_task(
                client.train_new(self.model_weights)))
        # wait for tasks to finish
        logging.info('waiting for tasks')
        for task in tasks:
            await task
        return tasks


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wait_task = None
        self.main_stream = None
        self.weight_streams = []
        self.new_model_weights = None

    def quic_event_received(self, event: QuicEvent) -> None:
        """
        Called when a QUIC event is received.
        Reimplement this in your subclass to handle the events.
        """
        if isinstance(event, ConnectionTerminated):
            print(event)
            reader = self._stream_readers.get(self.main_stream, None)
            if reader is not None:
                reader.feed_eof()
            if self._wait_task is not None and not self._wait_task.done():
                self._wait_task.cancel()
        elif isinstance(event, StreamDataReceived):
            reader = self._stream_readers.get(event.stream_id, None)
            if reader is None:
                reader = asyncio.StreamReader()
                self._stream_readers[event.stream_id] = reader
            reader.feed_data(event.data)
            if self.main_stream is None:
                self.main_stream = event.stream_id
        elif isinstance(event, HandshakeCompleted):
            logging.info(event)
            self.initiate_comunication()

    async def train_new(self, model_weights):
        '''One round of training for a client
        '''
        main_reader = self._stream_readers.get(self.main_stream)
        # End the wait task
        if self._wait_task is not None:
            logging.info('canjceling _wait_task')
            self._wait_task.cancel()
            try:
                await self._wait_task
            except asyncio.CancelledError:
                logging.info("_wait_task is cancelled for %s", self)

        # Signal send weights so client can build the receive weights tasks
        self.send(b'Sending Weights\n', self.main_stream)

        if not self.new_model_weights:
            self.new_model_weights = [None for x in model_weights]
        # create readers adn store idx-stream id
        if not self.weight_streams:
            for index in range(len(model_weights)):
                stream_id = self.create_new_stream()
                self.weight_streams.append(stream_id)
                self._stream_readers[stream_id] = asyncio.StreamReader(
                )
        # send info message
        info_message = []
        weights_binary = []
        for index, layer_weights in enumerate(model_weights):
            weight_binary = pickle.dumps(layer_weights)
            weights_binary.append(weight_binary)
            lenghth = str(len(weight_binary)).encode()
            stream_id = self.weight_streams[index]
            # IN case no stream for weights
            if stream_id is None:
                stream_id = self.create_stream()
                self._stream_readers[stream_id] = asyncio.StreamReader(
                )
            info_message.append(
                b':'.join([str(stream_id).encode(), lenghth]))
        self.send(b'|'.join(info_message)+b'\n', self.main_stream)

        print('Sending weights')
        # Send the weights
        received_tasks = []
        for index, weight in enumerate(weights_binary):
            stream_id = self.weight_streams[index]
            # create a task that will signal if the weights were received by server in Timeout
            task = self._loop.create_task(
                asyncio.wait_for(self.wait_weights_received(stream_id), 60))
            received_tasks.append(task)
            self.send(weight, stream_id)

        # signal if servwr received the weights
        for index, task in enumerate(received_tasks):
            try:
                await task
            except asyncio.TimeoutError:
                logging.info(
                    'Weights for layer %s were not received by client in time', index)
        # Signal client to train as all weights were sent
        self.send(b'Proceed to Training\n', self.main_stream)
        while True and not main_reader.at_eof():
            message = await main_reader.readline()
            if message == b'Finished training. Sending weights back.\n':
                sample_size_message = await main_reader.readline()
                sample_size = int(sample_size_message.split(b':')[1])
                info_message = await main_reader.readline()
                # s_id:length of bite array | ...
                receive_info = [x.split(b':')
                                for x in info_message.split(b'|')]
                for index, element in enumerate(receive_info):
                    stream_id = int(element[0])
                    reader = self._stream_readers.get(stream_id, None)
                    # if no reader for new stream
                    if reader is None:
                        reader = asyncio.StreamReader()
                        self._stream_readers[stream_id] = reader
                    self.weight_streams[index] = stream_id
                get_weights_tasks = [None for i in model_weights]
                for index in range(len(model_weights)):
                    stream_id = int(receive_info[index][0])
                    length = int(receive_info[index][1])
                    task = self._loop.create_task(asyncio.wait_for(
                        self.get_weights_for_layer(stream_id, length), 60))
                    get_weights_tasks[index] = task
                # Wait for all weights to be received in Timeout
                for index, task in enumerate(get_weights_tasks):
                    try:
                        logging.info(
                            'Waiting for task at index %s', str(index))
                        weight = await task
                        # asign new weights to
                        self.new_model_weights[index] = weight
                    # if weights are not received in timeout log
                    except asyncio.TimeoutError:
                        logging.info(
                            'Weights for layer %s were not received in time', index)
                return self.new_model_weights, sample_size

    async def get_weights_for_layer(self, stream_id, length):
        reader = self._stream_readers.get(stream_id)
        if reader is None:
            reader = asyncio.StreamReader()
            self._stream_readers[stream_id] = reader
        binary_weights = await reader.readexactly(length)
        # sende received
        self.send(b'Received Weights\n', stream_id)
        return pickle.loads(binary_weights)

    async def wait_weights_received(self, stream_id):
        reader = self._stream_readers.get(stream_id)
        while True:
            line = await reader.readline()
            print(line)
            if line == b'Received Weights\n':
                return

    def initiate_comunication(self):
        stream_id = self.create_new_stream()
        self.main_stream = stream_id
        self._stream_readers[stream_id] = asyncio.StreamReader()
        self._wait_task = self._loop.create_task(self.say_wait(stream_id))

    def create_wait_task(self):
        # reschedule wait
        self._wait_task = self._loop.create_task(
            self.say_wait(self.main_stream))

    async def say_wait(self, stream_id):
        while True:
            self.send(b'Wait for instructions\n', stream_id)
            await asyncio.sleep(10)

    def create_new_stream(self):
        stream_id = self._quic.get_next_available_stream_id()
        self._quic._get_or_create_stream_for_send(stream_id)
        return stream_id

    def send(self, data, stream_id, end_stream=False):
        self._quic.send_stream_data(stream_id, data, end_stream)
        self.transmit()


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
