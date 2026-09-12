import hypernetx as hnx
from hypernetx.algorithms import s_closeness_centrality

class SClosenessFeatureExtractor:
    def __init__(self):
        self.s_values = None
    def build_hypergraph(self, streaming_hg):
        hyperedges = streaming_hg.get_hyperedges()
        edge_dict = {
            ip: list(ports.keys())
            for ip, ports in hyperedges.items()
        }
        return hnx.Hypergraph(edge_dict)

    def get_max_hyperedge_size(self, streaming_hg):
        hyperedges = streaming_hg.get_hyperedges()

        if not hyperedges:
            return 0

        return max(
            len(ports)
            for ports in hyperedges.values()
        )

    def calculate_skip_interval(self, max_size):
        if max_size < 13:
            return None

        target = 0.70 * max_size
        k = round((target - 3) / 10)

        return max(k, 1)

    def calibrate(self, streaming_hg):
        max_size = self.get_max_hyperedge_size(streaming_hg)
        k = self.calculate_skip_interval(max_size)

        if k is None:
            self.s_values = []
        else:
            self.s_values = [
                3 + n * k
                for n in range(11)
            ]

        return self.s_values

    def compute_centrality(self, streaming_hg):
        if self.s_values is None:
            raise RuntimeError(
                "S-values have not been calibrated."
            )

        if not self.s_values:
            return {}

        H = self.build_hypergraph(streaming_hg)

        centrality = {}

        for s in self.s_values:
            centrality[s] = s_closeness_centrality(
                H,
                s=s,
                edges=True
            )

        return centrality

    def get_ip_features(self, centrality, ip):
        if not self.s_values:
            return []

        features = []

        for s in self.s_values:
            value = centrality[s].get(ip, 0.0)
            features.append(value)

        features.append(sum(features))

        return features