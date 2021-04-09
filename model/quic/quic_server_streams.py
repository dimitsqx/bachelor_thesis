import sys
import argparse
import asyncio
import importlib
import logging
import time
import gc
from timeit import default_timer as timer
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
from pathlib import Path

from fl_model import create_keras_model
try:
    import uvloop
except ImportError:
    uvloop = None

with open('model/quic/EMNINST_TEST_2NN.pickle', 'rb') as fp:
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

    async def train_model(self, client_protocols, min_clients=10):
        '''Function called by the server to train the model
        '''
        if len(self.performance_hystory) == 1:
            for client in self.performance_hystory[0]['client_info']:
                print(client['send_time'])
        if len(self.performance_hystory) == 300:
            with open('model/quic/history.pickle', 'wb') as fp:
                pickle.dump(self.performance_hystory, fp)
            with open('model/quic/server_weights', 'wb') as fp:
                pickle.dump(self.model_weights, fp)
            for client in client_protocols:
                client.close()
            sys.exit()
        if len(client_protocols) >= min_clients:
            history = {
                'client_info': [],
                'accuracy': None
            }
            # Sample min_clients
            train_clients = set(random.sample(client_protocols, min_clients))
            # print('Train clients')
            # print(train_clients)
            # Get remaining clients
            other_clients = client_protocols.difference(train_clients)
            # print('Other clients')
            # print(other_clients)
            # Train for number of rounds
            if train_clients.issubset(self.find_clients()):
                # Get the tasks for client
                fit_tasks = await self.fit_round(train_clients)
                terminated_tasks = []
                for task in fit_tasks:
                    if task.result() is not None:
                        #############################################################
                        # For multiple streams
                        # with x% chance drop the  layer of weights
                        # x = 0.5
                        # weights = []
                        # for layer in task.result()[0]:
                        #     drop = np.random.binomial(1, x)
                        #     if drop == 0:
                        #         weights.append(None)
                        #     else:
                        #         weights.append(layer)
                        # samples = task.result()[1]
                        # time_to_send = task.result()[2]
                        # terminated_tasks.append((weights, samples))
                        # For one stream
                        weights = task.result()[0]
                        samples = task.result()[1]
                        time_to_send = task.result()[2]
                        x = 0.5
                        drop = np.random.binomial(1, x)
                        if drop == 1 and samples is not None:
                            terminated_tasks.append((weights, samples))
                        #########################################################
                        history['client_info'].append({
                            # 'weights': weights,
                            # 'samples': samples,
                            'send_time': time_to_send
                        })
                    else:
                        history['client_info'].append(None)
                logging.info('Weights received back %s',
                             str(len(terminated_tasks)))
                # FEdAVG
                for index in range(len(self.model_weights)):
                    # index of  task that returned weights for this layer
                    task_returned_weights_idx = [i for i, e in enumerate(
                        terminated_tasks) if e[0][index] is not None]
                    # calculate sample size for tasks that returned weights for this layer
                    sample_size = sum(terminated_tasks[i][1]
                                      for i in task_returned_weights_idx)
                    logging.warning('Layer: %s', index)
                    logging.warning('Returned %s', len(
                        task_returned_weights_idx))
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

                self.fl_model.set_weights(self.model_weights)
                # evaluate model
                eval_results = self.fl_model.evaluate(X_TEST, Y_TEST)
                history['accuracy'] = eval_results
                print(eval_results)
                self.performance_hystory.append(history)
            else:
                logging.info(
                    'Finished on round because not all clients are available')

            self.finish_training(
                train_clients.intersection(self.find_clients()))
        else:
            with open('model/quic/history.pickle', 'wb') as fp:
                pickle.dump(self.performance_hystory, fp)

    def finish_training(self, clients):
        # print(clients)
        for client in clients:
            # client.close()
            client.create_wait_task()

    async def fit_round(self, clients):
        tasks = []
        # Create tasks for each client
        for client in clients:
            ########################################################
            # Multiple streams
            # tasks.append(self._loop.create_task(
            #     client.train_new(self.model_weights)))
            # One stream
            x = 1.0
            drop = np.random.binomial(1, x)
            if drop == 1:
                tasks.append(self._loop.create_task(
                    client.train_one_stream(self.model_weights)))
            ##################################################################
        # wait for tasks to finish
        logging.info('Only %s clients will receive the weights',
                     str(len(tasks)))
        for index, task in enumerate(tasks):
            logging.warning('Waiting client %s', index)
            await task
        return tasks


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wait_task = None
        self.main_stream = None
        self.weight_streams = []
        self.new_model_weights = None
        self.sent_time = None

    def quic_event_received(self, event: QuicEvent) -> None:
        """
        Called when a QUIC event is received.
        Reimplement this in your subclass to handle the events.
        """
        if isinstance(event, ConnectionTerminated):
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
            wait = self._wait_task
            self._wait_task = None
            wait.cancel()
            try:
                await wait
            except asyncio.CancelledError:
                logging.info("_wait_task is cancelled for %s", self)

        # Signal send weights so client can build the receive weights tasks
        self.send(b'Sending Weights\n', self.main_stream)

        self.new_model_weights = [None for x in model_weights]

        self.sent_time = [None for x in model_weights]
        self.receive_time = [None for x in model_weights]

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

        # print('Sending weights')
        # Send the weights
        received_tasks = []

        for index, weight in enumerate(weights_binary):
            stream_id = self.weight_streams[index]
            # create a task that will signal if the weights were received by server in Timeout
            task = self._loop.create_task(
                asyncio.wait_for(self.wait_weights_received(stream_id), 60))
            received_tasks.append(task)
            self.sent_time[index] = timer()
            self.send(weight, stream_id)

        # signal if servwr received the weights
        for index, task in enumerate(received_tasks):
            try:
                end = await task
                start = self.sent_time[index]
                self.sent_time[index] = (start, end)
            except asyncio.TimeoutError:
                # Set to None as not recceived
                self.sent_time[index] = None
                logging.warning(
                    'Weights for layer %s were not received by client in time', index)
        # TODO; if sent time all None return None

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
                        logging.warning(
                            'Weights for layer %s were not received in time', index)
                return self.new_model_weights, sample_size, self.sent_time

    async def train_one_stream(self, model_weights):
        '''use just one stream for weights'''
        main_reader = self._stream_readers.get(self.main_stream)
        # End the wait task
        if self._wait_task is not None:
            wait = self._wait_task
            self._wait_task = None
            wait.cancel()
            try:
                await wait
            except asyncio.CancelledError:
                logging.info("_wait_task is cancelled for %s", self)

        # Signal send weights so client can build the receive weights tasks
        self.send(b'Sending Weights\n', self.main_stream)

        self.new_model_weights = None

        self.sent_time = None
        self.receive_time = None

        weight_binary = pickle.dumps(model_weights)
        length = str(len(weight_binary)).encode()
        # Send the length
        self.send(length+b'\n', self.main_stream)
        # Send the weights
        start = timer()
        self.send(weight_binary, self.main_stream)
        try:
            end = await asyncio.wait_for(self.wait_weights_received(self.main_stream), 60)
        except asyncio.TimeoutError:
            logging.warning(
                'Weights for layer  were not received by client in time')
            return None
        # Signal client to train as all weights were sent
        self.send(b'Proceed to Training\n', self.main_stream)
        while True and not main_reader.at_eof():
            message = await main_reader.readline()
            if message == b'Finished training. Sending weights back.\n':
                sample_size_message = await main_reader.readline()
                sample_size = int(sample_size_message.split(b':')[1])
                info_message = await main_reader.readline()
                length = int(info_message)
                try:
                    self.new_model_weights = await asyncio.wait_for(
                        self.get_weights_for_layer(self.main_stream, length), 60)
                except asyncio.TimeoutError:
                    logging.warning(
                        'Weights for layer were not received in time')
                    return None

                return self.new_model_weights, sample_size, (start, end)

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
        # print(reader)
        # index = self.weight_streams.index(stream_id)
        while True:
            line = await reader.readline()
            # print(line)
            if line == b'Received Weights\n':
                end = timer()
                return end

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
            await asyncio.sleep(30)

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
