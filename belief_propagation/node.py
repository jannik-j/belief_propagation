from __future__ import annotations
import numpy as np
from typing import Any, Callable
from functools import total_ordering
from abc import ABC, abstractmethod


@total_ordering
class Node(ABC):
    """Base class VNodes anc CNodes.
    Derived classes are expected to implement an "initialize" and  method a "message" which should return the message to
    be passed on the graph.
    Nodes are ordered and deemed equal according to their ordering_key.
    """

    def __init__(self, uid: int, num_pcms: int, name: str = "") -> None:
        """
        :param name: name of node
        """
        self.uid = uid
        self.name = name if name else str(self.uid)
        self.ordering_key = str(self.uid)
        self.neighbors: list[dict[int, Node]] = [{} for _ in range(num_pcms)]  # keys as senders uid
        self.received_messages: dict[int, Any] = {}  # keys as senders uid, values as messages

    def register_neighbor(self, neighbor: Node, num_pcm: int) -> None:
        self.neighbors[num_pcm][neighbor.uid] = neighbor

    def __str__(self) -> str:
        if self.name:
            return self.name
        else:
            return str(self.uid)

    def get_neighbors(self, num_pcm: int) -> list[int]:
        return list(self.neighbors[num_pcm].keys())

    def receive_messages(self, num_pcm: int) -> None:
        for node_id, node in self.neighbors[num_pcm].items():
            self.received_messages[node_id] = node.message(self.uid)

    @abstractmethod
    def message(self, requester_uid: int) -> Any:
        pass

    @abstractmethod
    def initialize(self):
        pass

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.ordering_key == other.ordering_key

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.ordering_key < other.ordering_key


class CNode(Node):

    def initialize(self, boxplus_type: str):
        self.received_messages = {node_uid: 0 for node_uid in self.neighbors[0]}
        if boxplus_type == "exact":
            self.boxplus = self._boxplus_exact
        elif boxplus_type == "minsum":
            self.boxplus = self._minsum
        else:
            raise ValueError("Unknown boxplus type")
    
    def _boxplus_exact(self, q: np.ndarray):
        def phi(x):
            return -np.log(np.tanh(x/2))
        return np.prod(np.sign(q))*phi(np.sum(phi(np.absolute(q))))
    
    def _minsum(self, q: np.ndarray):
        return np.prod(np.sign(q))*np.absolute(q).min()

    def message(self, requester_uid: int) -> np.float_:
        q = np.array([msg for uid, msg in self.received_messages.items() if uid != requester_uid])
        return self.boxplus(q)


class VNode(Node):
    def __init__(self, uid: int, channel_model: Callable, num_pcms: int, name: str = ""):
        """
        :param channel_model: a function which receives channel outputs anr returns relevant message
        :param ordering_key: used to order nodes per their order in the parity check matrix
        :param name: optional name of node
        """
        self.channel_model = channel_model
        self.channel_symbol: int = None  # currently assuming hard channel symbols
        self.channel_llr: np.float_ = None
        super().__init__(uid, num_pcms, name)

    def initialize(self, channel_symbol):
        self.channel_symbol = channel_symbol
        self.channel_llr = self.channel_model(channel_symbol)
        self.received_messages = {node_uid: 0 for node_uid in self.neighbors[0]}

    def message(self, requester_uid: int) -> np.float_:
        return self.channel_llr + np.sum(
            [msg for uid, msg in self.received_messages.items() if uid != requester_uid]
        )

    def estimate(self) -> np.float_:
        return self.channel_llr + np.sum(list(self.received_messages.values()))
