from belief_propagation import BeliefPropagation, TannerGraph, bsc_llr
import numpy as np


# pcms = np.load("../code/arrays/latest_BCH_31_3272_rbow.npy")
pcm = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
              [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
              [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
              [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]])
pcm_1 = pcm.copy()
pcm_1[1] += pcm_1[2]
pcm_1[1] += pcm_1[3]
pcm_1[4] += pcm_1[0]
pcm_1[2] += pcm_1[4]
pcm_1 %= 2
pcms = np.stack([pcm, pcm_1], axis=0)

model = bsc_llr(0.1)
graph = TannerGraph(pcms, model)

# c = np.random.randint(0, 2, size=(31,))
# codeword [1,1,0,0,1,0,0,0,0,0]
c = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1])

bp = BeliefPropagation(graph, pcms, max_iter=100, num_reps=1, boxplus_type="minsum")
estimate, llr, decode_success = bp.decode(c)
print(f"Success: {decode_success}")
print(np.logical_xor(c, estimate))