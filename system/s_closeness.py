# import hypernetx as hnx
# from hypernetx.algorithms import s_closeness_centrality
# from hypergraph import StreamingHypergraph

# class SClosenessAnalyzer:

#     def __init__(self):
#         pass

#     def build_hypergraph(self, streaming_hg):
#         """
#         Convert the current StreamingHypergraph into
#         a HyperNetX hypergraph.

#         Each IP is a hyperedge.
#         Each destination port is a vertex.
#         """

#         hyperedges = streaming_hg.get_hyperedges()
#         edge_dict = {
#             ip: list(ports.keys())
#             for ip, ports in hyperedges.items()
#         }

#         return hnx.Hypergraph(edge_dict)

#     def get_max_hyperedge_size(self, streaming_hg):
#         """
#         Return the maximum number of destination ports
#         connected to any hyperedge.
#         """

#         hyperedges = streaming_hg.get_hyperedges()

#         if not hyperedges:
#             return 0

#         return max(
#             len(ports)
#             for ports in hyperedges.values()
#         )

#     def calculate_skip_interval(self, max_size):
#         """
#         Calculate k according to the strategy used
#         in the base paper.

#         3 + 10k should be approximately 70%
#         of the maximum hyperedge size.
#         """

#         if max_size < 13:
#             return None

#         target = 0.70 * max_size

#         k = round((target - 3) / 10)

#         return max(k, 1)

#     def get_s_values(self, max_size):

#         k = self.calculate_skip_interval(max_size)

#         if k is None:
#             return []

#         return [
#             3 + n * k
#             for n in range(11)
#         ]

#     def compute(self, streaming_hg):

#         H = self.build_hypergraph(streaming_hg)

#         max_size = self.get_max_hyperedge_size(
#             streaming_hg
#         )

#         k = self.calculate_skip_interval(max_size)

#         if k is None:
#             return {
#                 "max_hyperedge_size": max_size,
#                 "skip_interval": None,
#                 "s_values": [],
#                 "centrality": {}
#             }

#         s_values = self.get_s_values(max_size)

#         results = {}

#         for s in s_values:

#             centrality = s_closeness_centrality(
#                 H,
#                 s=s,
#                 edges=True
#             )

#             results[s] = centrality

#         return {
#             "max_hyperedge_size": max_size,
#             "skip_interval": k,
#             "s_values": s_values,
#             "centrality": results
#         }