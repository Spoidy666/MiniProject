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

    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    # WINDOW-ADAPTIVE S-CLOSENESS CENTRALITY FOR STREAMING INFERENCE
    # Re-calibrates s-values proportionally to the current window's hypergraph
    # size while keeping results keyed by training s-values so feature column
    # names match the saved model. This fixes the "s-value mismatch" where
    # training s-values (e.g. s=1163) are too large for small 5-minute windows
    # (max edge size ~20-50), causing 78%+ of features to be zero.
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    def compute_adaptive_centrality(self, streaming_hg):
        if self.s_values is None or not self.s_values:
            return {}, []

        max_size = self.get_max_hyperedge_size(streaming_hg)

        if max_size < 2:
            # Graph too small for any meaningful s-closeness
            return {s: {} for s in self.s_values}, [0] * len(self.s_values)

        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
        # GENERATE WINDOW-PROPORTIONAL S-VALUES SCALED TO CURRENT GRAPH SIZE
        # Uses the same calibration formula but based on the window's max edge
        # size instead of the full training graph. This ensures s-values are
        # computable within the window (never exceeding max_size).
        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
        if max_size < 13:
            # Small window: distribute s-values linearly from 1 to max_size
            n_vals = len(self.s_values)
            window_s_values = [
                max(1, int(round(1 + i * (max_size - 1) / max(n_vals - 1, 1))))
                for i in range(n_vals)
            ]
        else:
            target = 0.70 * max_size
            k = max(round((target - 3) / 10), 1)
            window_s_values = [3 + n * k for n in range(len(self.s_values))]

        # Cap all window s-values at max_size (cannot exceed actual graph)
        window_s_values = [min(s, max_size) for s in window_s_values]
        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

        H = self.build_hypergraph(streaming_hg)

        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
        # COMPUTE CENTRALITY AT WINDOW-LOCAL S-VALUES, KEYED BY TRAINING S-VALUES
        # The centrality dict uses training s-values as keys so that
        # get_ip_features() and HGIFeatureBuilder produce columns matching
        # the saved LightGBM model's expected feature names.
        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
        centrality = {}
        for training_s, window_s in zip(self.s_values, window_s_values):
            centrality[training_s] = s_closeness_centrality(
                H,
                s=window_s,
                edges=True
            )

        return centrality, window_s_values
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

    def get_ip_features(self, centrality, ip):
        if not self.s_values:
            return []

        features = []

        for s in self.s_values:
            value = centrality[s].get(ip, 0.0)
            features.append(value)

        features.append(sum(features))

        return features