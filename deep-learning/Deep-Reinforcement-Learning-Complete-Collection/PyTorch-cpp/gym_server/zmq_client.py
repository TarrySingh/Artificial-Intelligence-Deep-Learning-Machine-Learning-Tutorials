"""
Pytorch-cpp-rl OpenAI gym server ZMQ client.
"""
import zmq
import msgpack


class ZmqClient:
    """
    Provides a ZeroMQ interface for communicating with client.
    """

    def __init__(self, port: int):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{port}")

    def receive(self) -> bytes:
        """
        Gets a message from the client.
        Blocks until a message is received.
        """
        message = self.socket.recv()
        try:
            response = msgpack.unpackb(message, raw=False)
        except msgpack.exceptions.ExtraData:
            response = message
        return response

    def send(self, message: object):
        """
        Sends a message to the client.
        """
        if isinstance(message, str):
            self.socket.send_string(message)
        else:
            self.socket.send(message.to_msg())
