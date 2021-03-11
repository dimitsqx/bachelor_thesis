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
        self._weights_waiter = None
        self.training_task = None
        self.model = create_keras_model()
        self.model_weights = self.model.get_weights()
        self.main_stream = None
        self.weight_streams = [None for x in self.model_weights]
        self.send_weight_task = [None for x in self.model_weights]
        self.get_weight_task = [None for x in self.model_weights]

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

    def handle_message(self, message: bytes, stream_id):

        if self.main_stream is None:
            self.main_stream = stream_id

        messages = message.split(b':')

        if messages[0] == b'Wait for instructions\n':
            return logging.info('Waiting')

        if messages[0] == b'Sending Weights\n' and self.training_task is None:
            logging.info('Waiting for weights')
            # Create tasks to receive weights
            # Create event that will lock access to the receive tasks untill they are built
            self._weights_waiter = self._loop.create_future()
            # Task to get the weights
            for index in range(len(self.get_weight_task)):
                awaitable = self._loop.create_future()
                task = self._loop.create_task(asyncio.wait_for(
                    asyncio.shield(awaitable), 60))
                self.get_weight_task[index] = (awaitable, task)
            # allow access to get weight tasks
            return self._weights_waiter.set_result(None)
        # if training is not in progress
        if messages[0] == b'Weights' and self.training_task is None and self._weights_waiter is not None:
            # Wait for get weights task to build
            self._loop.run_until_complete(asyncio.shield(self._weights_waiter))
            # get new weights
            awaitable, task = self.get_weight_task[int(messages[1].decode())]
            awaitable.set_result(pickle.loads(messages[2]))
            # set the layer weights
            self.model_weights[int(messages[1].decode())
                               ] = pickle.loads(messages[2])
            # store the stream id for layer
            self.weight_streams[int(messages[1].decode())] = stream_id
            return self.send(b'Received Weights\n', stream_id)

        if messages[0] == b'Received Weights\n':
            # get the layer index for this stream
            wait_task_index = self.weight_streams.index(stream_id)
            # get the waiter and task
            awaitable, task = self.send_weight_task[wait_task_index]
            # signal that the weights were received
            return awaitable.set_result(None)

        if messages[0] == b'Proceed to Training\n' and self._weights_waiter is not None:
            # Wait for all weights to be received in Timeout
            for index, (awaitable, task) in enumerate(self.get_weight_task):
                try:
                    self._loop.run_until_complete(task)
                # if weights are not received in timeout log
                except asyncio.TimeoutError:
                    self.get_weight_task[index] = None
                    logging.info(
                        'Weights for layer %s were not received', index)
            # Reset weights waiter so no new weights can be received
            self._weights_waiter = None
            # Create a task THAT SIGNALS TRAINING
            self.training_task = self._loop.create_task(
                self.say_in_training())
            # TRAIN FOR ONE ROUND
            self.train_one_round()
            # Signal the training has finished
            self.send(b'Finished training. Sending weights back.\n',
                      self.main_stream)
            # Send the weights
            for index, weight in enumerate(self.model_weights):
                stream_id = self.weight_streams[index]
                # IN case no stream for weights not received
                if stream_id is None:
                    stream_id = self.create_stream()
                message = b':'.join(
                    [b'Weights', str(index).encode(), pickle.dumps(self.model_weights[index])])
                # create a task that will signal if the weights were received by server in Timeout
                awaitable = self._loop.create_future()
                task = self._loop.create_task(
                    asyncio.wait_for(asyncio.shield(awaitable), 60))
                self.send_weight_task[index] = (awaitable, task)
                # send weights
                self.send(message, stream_id)

            # Cancel task that signals training
            self.training_task.cancel()
            try:
                self._loop.run_until_complete(self.training_task)
            except asyncio.CancelledError:
                logging.info("Training task is cancelled for %s", self)
            # Wait for wait received from server
            for index, (awaitable, task) in enumerate(self.send_weight_task):
                try:
                    self._loop.run_until_complete(task)
                    self.send_weight_task[index] = None
                except asyncio.TimeoutError:
                    self.send_weight_task[index] = None
                    logging.info(
                        'Weights for layer %s were not received by server', index)

    def train_one_round(self):
        self.model.set_weights(self.model_weights)
        # Load data
        x_train, y_train = DATA[next(KEYS)]
        # Start training
        self.model.fit(x_train, y_train, epochs=15,
                       batch_size=40, validation_split=0.2)

        self.model_weights = self.model.get_weights()

        return self.send(b':'.join([b'Sample', '{0}'.format(len(x_train)).encode()]), self.main_stream)

    async def say_in_training(self):
        while True:
            self.send(b'In training\n', self.main_stream)
            await asyncio.sleep(10)

    def create_stream(self):
        stream_id = self._quic.get_next_available_stream_id()
        self._quic._get_or_create_stream_for_send(stream_id)
        return stream_id

    def send(self, data, stream_id, end_stream=False):
        self._quic.send_stream_data(stream_id, data, end_stream)
        self.transmit()


def save_session_ticket(ticket: SessionTicket) -> None:
    """
    Callback which is invoked by the TLS engine when a new session ticket
    is received.
    """
    logging.info("New session ticket received")
    if session_ticket_path:
        with open(session_ticket_path, "wb") as fp:
            pickle.dump(ticket, fp)


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
