import pandas as pd

from hypergraph import StreamingHypergraph
from dataManipulation import (
    clean_dataset,
    prepare_labels,
    prepare_data
)
from timeWindow import TimeWindow
from sClosenessFeatureExtractor import SClosenessFeatureExtractor
from hgiFeatureBuilder import HGIFeatureBuilder


CSV_PATH = r"..\dataset\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"


def main():

    # --------------------------------------------------
    # 1. Load and prepare dataset
    # --------------------------------------------------

    dataset = pd.read_csv(CSV_PATH)

    df = clean_dataset(dataset)
    df = prepare_labels(df)

    stream_data, hypergraph_data, metadata, X_nrf = prepare_data(df)

    # --------------------------------------------------
    # 2. Initialize streaming components
    # --------------------------------------------------

    hg = StreamingHypergraph()

    window_size = pd.Timedelta(minutes=5)
    time_window = TimeWindow(5)

    extractor = SClosenessFeatureExtractor()

    feature_builder = None

    first_analysis = True

    # Flows belonging to the current analysis interval
    current_window_indices = []

    # --------------------------------------------------
    # 3. Start temporal stream
    # --------------------------------------------------

    first_timestamp = hypergraph_data["Timestamp"].iloc[0]

    time_window.start(first_timestamp)

    for index, row in hypergraph_data.iterrows():

        source_ip = row["Source IP"]
        destination_ip = row["Destination IP"]
        destination_port = row["Destination Port"]
        timestamp = row["Timestamp"]

        # ----------------------------------------------
        # Add flow to temporal hypergraph
        # ----------------------------------------------

        hg.add_flow(
            source_ip,
            destination_ip,
            destination_port,
            timestamp
        )

        # ----------------------------------------------
        # Remove expired relationships
        # ----------------------------------------------

        hg.remove_expired_flows(
            timestamp,
            window_size
        )

        # Keep track of this flow
        current_window_indices.append(index)

        # ----------------------------------------------
        # Check whether analysis window completed
        # ----------------------------------------------

        if time_window.is_complete(timestamp):

            print(
                "\nAnalysis window completed at:",
                timestamp
            )

            # ------------------------------------------
            # First window = structural calibration
            # ------------------------------------------

            if first_analysis:

                s_values = extractor.calibrate(hg)

                print(
                    "Calibrated s-values:",
                    s_values
                )

                feature_builder = HGIFeatureBuilder(
                    extractor,
                    ip_column="Destination IP"
                )

                first_analysis = False

            # ------------------------------------------
            # Calculate current HG centrality
            # ------------------------------------------

            centrality = extractor.compute_centrality(hg)

            # ------------------------------------------
            # Extract data belonging to this window
            # ------------------------------------------

            window_stream_data = stream_data.loc[
                current_window_indices
            ]

            window_nrf = X_nrf.loc[
                current_window_indices
            ]

            window_metadata = metadata.loc[
                current_window_indices
            ]

            # ------------------------------------------
            # Build HGI features
            # ------------------------------------------

            X_hgi = feature_builder.build_features(
                window_stream_data,
                window_nrf,
                centrality
            )

            print(
                "HGI feature shape:",
                X_hgi.shape
            )

            print(
                "First HGI vector:"
            )

            print(
                X_hgi.iloc[0].to_dict()
            )

            # ------------------------------------------
            # Reset for next temporal window
            # ------------------------------------------

            current_window_indices = []

            time_window.reset(timestamp)

    # --------------------------------------------------
    # 4. Final state
    # --------------------------------------------------

    print("\nFinal hypergraph:")

    print(
        "Hyperedges:",
        hg.number_of_hyperedges()
    )

    print(
        "Vertices:",
        hg.number_of_vertices()
    )


if __name__ == "__main__":
    main()