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

    async def train_model(self, client_protocols, num_rounds=2, min_clients=1):
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
                                        for task in fit_tasks if task.result()[1] is not None]
                    print('Training round finished')
                    print(terminated_tasks)
                    print(self.model_weights)
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
                            print(type(weights))
                            samples = terminated_tasks[idx][1]
                            print(samples)
                            if layer is not None:
                                layer += (weights*samples)/sample_size
                            else:
                                layer = (weights*samples)/sample_size
                                print(layer)
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
                client.train(self.model_weights)))
        # wait for tasks to finish
        logging.info('waiting for tasks')
        for task in tasks:
            await task
        return tasks


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_transport = dict()
        self._ack_waiter = dict()
        self.wait_weights = None
        self._wait_task = None
        self.main_stream = None
        self.weight_streams = []
        self.send_weight_task = []
        self.get_weight_task = []
        self.wait_sample_size = None
        self.new_model_weights = None

    def quic_event_received(self, event: QuicEvent) -> None:
        """
        Called when a QUIC event is received.
        Reimplement this in your subclass to handle the events.
        """
        if isinstance(event, ConnectionTerminated):
            print(event)
            if self._wait_task is not None and not self._wait_task.done():
                self._wait_task.cancel()
            if self.wait_weights is not None and not self.wait_weights.done():
                self.wait_weights.set_result(None)
            elif self.wait_weights is None:
                self.wait_weights = self._loop.create_future()
                self.wait_weights.set_result(None)
            if self.wait_sample_size is not None and not self.wait_sample_size.done():
                self.wait_sample_size.set_result(None)
            elif self.wait_sample_size is None:
                self.wait_sample_size = self._loop.create_future()
                self.wait_sample_size.set_result(None)
            for tmp in self.send_weight_task:
                if tmp is not None and not tmp[0].done():
                    awaitable = tmp[0]
                    awaitable.set_result(None)
            for tmp in self.get_weight_task:
                if tmp is not None and not tmp[0].done():
                    awaitable = tmp[0]
                    awaitable.set_result(None)

        elif isinstance(event, StreamDataReceived):
            # print(event)
            self.message_handler(event.data, event.stream_id)

        elif isinstance(event, HandshakeCompleted):
            logging.info(event)
            self.initiate_comunication()

    def message_handler(self, message, stream_id):
        if message == b'Received Weights\n':
            # get the layer sent on this stream_id
            wait_task_index = self.weight_streams.index(stream_id)
            # get the waiter for this layer
            (awaitable, task) = self.send_weight_task[wait_task_index]
            # Signal that leyer results were received by client
            return awaitable.set_result(None)

        if message == b'Finished training. Sending weights back.\n':
            if self.wait_weights is not None:
                # Create task to wait for each layer weights
                logging.info('Waiting for weights')
                for index in range(len(self.get_weight_task)):
                    awaitable = self._loop.create_future()
                    task = self._loop.create_task(asyncio.wait_for(
                        asyncio.shield(awaitable), 60))
                    self.get_weight_task[index] = (awaitable, task)
                # signal get_weights_task was created
                return self.wait_weights.set_result(None)
            return

        messages = message.split(b':')
        if messages[0] == b'Sample' and self.wait_sample_size is not None:
            # set the sample size
            self.wait_sample_size.set_result(int(messages[1].decode()))

        if messages[0] == b'Weights' and self.wait_weights is not None:
            # wait for get wait tasks to build
            if not self.wait_weights.done():
                self._loop.create_task(self.wait_weights)

            # if message als ocontains end
            if messages[-1] == b'End Weights':
                self.new_model_weights[int(messages[1].decode())] = b':'.join(
                    messages[2:-1])
                # signal received
                awaitable, task = self.get_weight_task[int(
                    messages[1].decode())]
                self.new_model_weights[int(messages[1].decode())] = pickle.loads(
                    self.new_model_weights[int(messages[1].decode())])
                awaitable.set_result(
                    self.new_model_weights[int(messages[1].decode())])
                self.send(b'Received Weights\n', stream_id)
            else:
                self.new_model_weights[int(messages[1].decode())] = b':'.join(
                    messages[2:])
            return

        # When weights are send through multiple messages
        if stream_id in self.weight_streams and self.wait_weights is not None:
            # get the layer index
            layer = self.weight_streams.index(stream_id)

            # if the receiving of data is not done
            if not self.get_weight_task[layer][0].done():
                # if ende of weights signal
                if messages[-1] == b'End Weights':
                    self.new_model_weights[layer] += b':'.join(messages[:-1])
                    awaitable, task = self.get_weight_task[layer]
                    self.new_model_weights[layer] = pickle.loads(
                        self.new_model_weights[layer])
                    awaitable.set_result(self.new_model_weights[layer])
                    self.send(b'Received Weights\n', stream_id)
                else:
                    self.new_model_weights[layer] += b':'.join(messages)
            return

    async def train(self, model_weights):
        '''One round of training for a client
        '''
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

        # Build
        if not self.weight_streams:
            self.weight_streams = [self.create_new_stream()
                                   for x in range(len(model_weights))]
        if not self.send_weight_task:
            self.send_weight_task = [None for x in model_weights]
        if not self.get_weight_task:
            self.get_weight_task = [None for x in model_weights]
        if not self.new_model_weights:
            self.new_model_weights = [None for x in model_weights]
        print(self.weight_streams)
        # send the weights
        for index, weights in enumerate(model_weights):
            # Create the message containing layer weights
            message = b':'.join(
                [b'Weights', str(index).encode(), pickle.dumps(weights), b'End Weights'])
            # create a task that will signal if the weights were received by clients in Timeout
            awaitable = self._loop.create_future()
            task = self._loop.create_task(
                asyncio.wait_for(asyncio.shield(awaitable), 60))

            self.send_weight_task[index] = (awaitable, task)
            # Send the message
            self.send(message, self.weight_streams[index])

        # wait for the client response on weights received
        for index, tmp in enumerate(self.send_weight_task):
            print(index)
            # if client did not send ack in timeout log it
            try:
                await tmp[1]
                self.send_weight_task[index] = None
            except asyncio.TimeoutError:
                self.send_weight_task[index] = None
                logging.info(
                    'Weights for layer %s were not received by client in time', index)

        # Signal client to train as all weights were sent
        self.send(b'Proceed to Training\n', self.main_stream)
        print('signalled go to training')
        # Create wait for weights signal used to create get_weights_tasks
        if self.wait_weights is None:
            self.wait_weights = self._loop.create_future()
        # Create wait for sample size
        if self.wait_sample_size is None:
            self.wait_sample_size = self._loop.create_future()
        # Wait for send weights message from server and create get weights tasks
        await asyncio.shield(self.wait_weights)
        print('12123123')
        # Wait for all weights to be received back
        for index, tmp in enumerate(self.get_weight_task):
            # if no tasks means connection dropped so skip
            if tmp is not None:
                try:
                    weights = await tmp[1]
                    self.new_model_weights[index] = weights
                except asyncio.TimeoutError:
                    self.get_weight_task[index] = None
                    self.new_model_weights[index] = None
                    logging.info(
                        'Weights for layer %s were not received', index)
        # Reset weights wait
        self.wait_weights = None
        # Wait for sample size
        sample_size = await asyncio.shield(self.wait_sample_size)
        print('123123123')
        self.wait_sample_size = None

        return self.new_model_weights, sample_size

    def initiate_comunication(self):
        stream_id = self.create_new_stream()
        self.main_stream = stream_id
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
