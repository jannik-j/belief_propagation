from .graph import TannerGraph
from numpy.typing import ArrayLike, NDArray
import numpy as np


class BeliefPropagation:
    def __init__(self, graph: TannerGraph, pcms: NDArray, max_iter: int, num_reps: int, boxplus_type: str = "exact"):
        self.pcm = pcms[0]
        self.num_pcms = pcms.shape[0]
        self.num_reps = num_reps
        self.graph = graph
        self.n = len(graph.v_nodes)
        self.max_iter = max_iter
        self.boxplus_type = boxplus_type

    def decode(self, channel_word) -> "tuple[NDArray, NDArray, bool]":
        if len(channel_word) != self.n:
            raise ValueError("incorrect block size")

        # initial step
        for idx, node in enumerate(self.graph.ordered_v_nodes()):
            node.initialize(channel_word[idx])
        for node in self.graph.c_nodes.values():  # send initial channel based messages to check nodes
            node.initialize(self.boxplus_type)

        for iter in range(self.max_iter):
            pcm_num = (iter // self.num_reps) % self.num_pcms
            print(pcm_num)
            # Variable to Check Node Step(vertical step)
            for node in self.graph.c_nodes.values():
                node.receive_messages(pcm_num)
            # Check to Variable Node Step(horizontal step):
            for node in self.graph.v_nodes.values():
                node.receive_messages(pcm_num)

            # Check stop condition
            llr = np.array([node.estimate() for node in self.graph.ordered_v_nodes()])
            estimate = np.array([1 if node_llr < 0 else 0 for node_llr in llr])
            syndrome = (self.pcm @ estimate) % 2
            if not syndrome.any():
                break

        return estimate, llr, not syndrome.any()
