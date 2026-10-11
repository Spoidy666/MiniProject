import pandas as pd
class HGIFeatureBuilder:
    NRF_COLUMNS = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Down/Up Ratio",
        "Protocol"
    ]

    def __init__(self, extractor, ip_column="Destination IP"):
        self.extractor = extractor
        self.ip_column = ip_column

    def get_feature_names(self):
        names = self.NRF_COLUMNS.copy()
        names.extend([f"s_closeness_{s}" for s in self.extractor.s_values
            ]
        )
        names.append("s_closeness_sum")
        return names

    def build_features(self, flows, nrf_data, centrality):
        if self.extractor.s_values is None:
            raise RuntimeError(
                "S-values have not been calibrated."
            )
        if len(flows) != len(nrf_data):
            raise ValueError(
                "flows and nrf_data must contain "
                "the same number of rows."
            )
        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
        # OPTIMIZED FEATURE EXTRACTION WITH IP CACHE
        # Pre-computes 12 hypergraph features per unique IP and fast-merges with NRF rows
        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
        unique_ips = flows[self.ip_column].unique()
        ip_cache = {
            ip: self.extractor.get_ip_features(centrality, ip)
            for ip in unique_ips
        }

        ips = flows[self.ip_column].tolist()
        nrf_list = nrf_data.values.tolist()

        feature_rows = [
            nrf_row + ip_cache[ip]
            for nrf_row, ip in zip(nrf_list, ips)
        ]
        # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

        return pd.DataFrame(
            feature_rows,
            columns=self.get_feature_names()
        )