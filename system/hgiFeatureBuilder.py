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
        feature_rows = []
        for i in range(len(flows)):

            # Get the IP from the DataFrame row
            ip = flows.iloc[i][self.ip_column]

            # Get the s-closeness features for this IP
            hg_features = self.extractor.get_ip_features(
                centrality,
                ip
            )

            # Get the 9 NRF features
            nrf_features = nrf_data.iloc[i].tolist()

            # 9 NRF + 11 s-closeness + 1 sum = 21
            row = nrf_features + hg_features

            feature_rows.append(row)

        return pd.DataFrame(
            feature_rows,
            columns=self.get_feature_names()
        )