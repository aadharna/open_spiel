from enum import Enum


class MessageTypes(Enum):
    """Message types for the queue."""

    # A message to be sent to the queue.
    QUEUE_MESSAGE = 0
    # A message to be sent to the queue to indicate that the queue should be
    # closed.
    QUEUE_CLOSE = 1
    # A message to be sent from the queue to indicate that the queue has been
    # closed.
    QUEUE_CLOSED = 2
    # A heartbeat message to be sent from the queue to indicate that the queue is
    # still alive.
    QUEUE_HEARTBEAT = 3
    # A message to be sent to the queue to indicate that the thread should send analysis data.
    QUEUE_ANALYSIS = 4


class QueueMessage(object):
    """A message to be sent to a queue."""

    def __init__(self, message_type, data=None):
        self.message_type = message_type
        self.data = data

    def __repr__(self):
        return "QueueMessage({}, {})".format(self.message_type, self.data)

    def __iter__(self):
        yield self.message_type
        yield self.data
