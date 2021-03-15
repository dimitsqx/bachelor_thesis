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
with open('model/quic/EMNINST_TRAIN.pickle', 'rb') as fp:
    DATA = pickle.load(fp)
    KEYS = iter(DATA)


class NewProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_task = None
        self.model = create_keras_model()
        self.model_weights = self.model.get_weights()
        self.main_stream = None
        self.weight_streams = [None for x in self.model_weights]

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
        elif isinstance(event, StreamDataReceived):
            reader = self._stream_readers.get(event.stream_id, None)
            if reader is None:
                reader = asyncio.StreamReader()
                self._stream_readers[event.stream_id] = reader
            reader.feed_data(event.data)
            if self.main_stream is None:
                self.main_stream = event.stream_id
                self._loop.create_task(self.communication_handler())

        elif isinstance(event, HandshakeCompleted):
            logging.info(event)

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
            if line == b'Received Weights\n':
                return

    async def communication_handler(self):
        main_reader = self._stream_readers.get(self.main_stream)
        while True and not main_reader.at_eof():
            message = await main_reader.readline()
            if message == b'Wait for instructions\n':
                logging.info('Waiting')
            if message == b'Sending Weights\n' and self.training_task is None:
                logging.info('Waiting for weights')
                # wait for info about lengts and stream _id for each index
                message = await main_reader.readline()
                # s_id:length of bite array | ...
                receive_info = [x.split(b':') for x in message.split(b'|')]
                for index, element in enumerate(receive_info):
                    stream_id = int(element[0])
                    reader = self._stream_readers.get(stream_id, None)
                    if reader is None:
                        reader = asyncio.StreamReader()
                        self._stream_readers[stream_id] = reader
                    self.weight_streams[index] = stream_id
                get_weights_tasks = [None for i in self.model_weights]
                for index in range(len(self.model_weights)):
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
                        print(weight)
                        # asign new weights to
                        self.model_weights[index] = weight
                    # if weights are not received in timeout log
                    except asyncio.TimeoutError:
                        logging.info(
                            'Weights for layer %s were not received in time', index)

            if message == b'Proceed to Training\n' and self.training_task is None:
                logging.info('proceding to training')
                # Create a task THAT SIGNALS TRAINING
                self.training_task = self._loop.create_task(
                    self.say_in_training())
                # TRAIN FOR ONE ROUND
                sample_size = self.train_one_round()
                # Signal the training has finished
                self.send(b'Finished training. Sending weights back.\n',
                          self.main_stream)
                # send sample size
                self.send(b':'.join([b'Sample', '{0}'.format(
                    sample_size).encode(), b'\n']), self.main_stream)
                # send info message
                info_message = []
                weights_binary = []
                for index, layer_weights in enumerate(self.model_weights):
                    weight_binary = pickle.dumps(layer_weights)
                    weights_binary.append(weight_binary)
                    lenghth = str(len(weight_binary)).encode()
                    stream_id = self.weight_streams[index]
                    # IN case no stream for weights not received
                    if stream_id is None:
                        stream_id = self.create_stream()
                        self._stream_readers[stream_id] = asyncio.StreamReader(
                        )
                        self.weight_streams[index] = stream_id
                    info_message.append(
                        b':'.join([str(stream_id).encode(), lenghth]))
                self.send(b'|'.join(info_message)+b'\n', self.main_stream)

                print('Sending weights back')
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
                            'Weights for layer %s were not received by server in time', index)

                print('Cancel signal in training')
                # Cancel task that signals training
                self.training_task.cancel()
                # log training_signal_cancel
                try:
                    await self.training_task
                except asyncio.CancelledError:
                    logging.info("Training task is cancelled for %s", self)
                self.training_task = None

    def train_one_round(self):
        print(self.model_weights)
        self.model.set_weights(self.model_weights)
        # Load data
        x_train, y_train = DATA[next(KEYS)]
        # Start training
        self.model.fit(x_train, y_train, epochs=15,
                       batch_size=40, validation_split=0.2)

        self.model_weights = self.model.get_weights()

        return len(x_train)

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
    # pylint: disable=not-async-context-manager
    async with connect(
        host,
        port,
        configuration=configuration,
        create_protocol=NewProtocol,
        session_ticket_handler=save_session_ticket,
    ) as client:
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
