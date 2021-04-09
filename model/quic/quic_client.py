import argparse
import asyncio
import logging
import os
import pickle
import ssl
import time
import tensorflow as tf
import numpy as np
from collections import deque
from typing import BinaryIO, Callable, Deque, Dict, List, Optional, Union, cast, Tuple
from urllib.parse import urlparse
import socket
import random
import wsproto
import wsproto.events
from quic_logger import QuicDirectoryLogger
from pathlib import Path
import aioquic
from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
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
with open('model/quic/EMNINST_TRAIN.pickle', 'rb') as fp:
    DATA = pickle.load(fp)
    CLIENTS = list(DATA.keys())


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


def save_session_ticket(ticket: SessionTicket) -> None:
    """
    Callback which is invoked by the TLS engine when a new session ticket
    is received.
    """
    logging.info("New session ticket received")
    if session_ticket_path:
        with open(session_ticket_path, "wb") as fp:
            pickle.dump(ticket, fp)


async def train_one_round(model, reader, writer, client):
    model_weights = await get_model_weights(reader, writer)
    model.set_weights(model_weights)
    # Load data
    x_train, y_train = None, None
    for key in client:
        if x_train is None and y_train is None:
            x_train, y_train = DATA[key]
        else:
            x, y = DATA[key]
            np.concatenate((x_train, x), axis=0)
            np.concatenate((y_train, y), axis=0)
# Start training
    writer.write(b'Start model train\n')
    model.fit(x_train, y_train, epochs=10, batch_size=40,
              validation_split=0.2, verbose=0)
    writer.write(b'Finished Round\n')
    # Wait for server to ack
    line = await reader.readline()
    if line == b'Send new weights\n':
        sent_weights = await send_model_weights(model.get_weights(), reader, writer)
        print('send sample size')
        writer.write('{0}\n'.format(len(x_train)).encode())


async def connect_client(configuration: QuicConfiguration, host: str, port: int, client):
    # pylint: disable=not-async-context-manager
    async with connect(
        host,
        port,
        configuration=configuration,
        create_protocol=NewProtocol,
        session_ticket_handler=save_session_ticket,
    ) as client:
        reader, writer = await client.create_stream()
        # while True:
        #     line= await reader.readline()
        #     if line==b'Prepare for training\n':
        #         break
        model = create_keras_model()
        writer.write(b'Ready to work\n')
        # print(client._quic.host_cid)
        # print(client._quic._peer_cid)

        while not reader.at_eof():
            line = await reader.readline()
            print(line)
            if line == b'Proceed to Training\n':
                tmp = await train_one_round(model, reader, writer, client)
        writer.write_eof()
        # print('sent eof')
        print((await reader.read()))
        client.close()
        await client.wait_closed()
        print('closed conn')


async def run(configuration: QuicConfiguration, host: str, port: int, index=0) -> None:
    try:
        with tf.device('cpu:0'):
            loop = asyncio.get_event_loop()
            clients = []
            for i in range(100):
                path = 'model/quic/client_weights/{}'.format(i)
                f = Path(path)
                if f.is_file():
                    print(i)
                    with open(path, 'rb') as fp:
                        client = pickle.load(fp)
                else:
                    if i == 99:
                        client = {
                            'index': i,
                            'weights': None,
                            'ids': CLIENTS[i*33:]
                        }
                    else:
                        client = {
                            'index': i,
                            'weights': None,
                            'ids': CLIENTS[i*33:(i+1)*33]
                        }
                clients.append(client)
            # clients = set(CLIENTS)
            i = index
            tasks = []
            # while True:
            # cl = random.sample(clients, 350)
            # used_clients.update(cl)
            for client in clients:
                task = loop.create_task(connect_client(
                    configuration, host, port, client))
                tasks.append(task)
            print(len(tasks))
            for task in tasks:
                await task
            del tasks[:]
            print('finished round {}'.format(i))
            i += 1
    except:
        for task in tasks:
            task.cancel()
        del tasks[:]
        return i

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

    loop = asyncio.get_event_loop()

    task = loop.run_until_complete(
        run(
            configuration=configuration,
            host='192.168.1.129',
            port=4433,
            index=0
        )
    )
