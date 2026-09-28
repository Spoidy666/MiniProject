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
from hgi_model import train_hgi_model

CSV_PATH = r"..\dataset\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"


def main():
    dataset = pd.read_csv(CSV_PATH)

    df = clean_dataset(dataset)
    df = prepare_labels(df)

    stream_data, hypergraph_data, metadata, X_nrf = prepare_data(df)

    # temporal 80/20 split
    split_index = int(len(df) * 0.8)

    train_indices = df.index[:split_index]
    stream_indices = df.index[split_index:]

    train_hypergraph_data = hypergraph_data.loc[train_indices]
    stream_hypergraph_data = hypergraph_data.loc[stream_indices]

    train_stream_data = stream_data.loc[train_indices]
    stream_stream_data = stream_data.loc[stream_indices]

    train_nrf = X_nrf.loc[train_indices]
    stream_nrf = X_nrf.loc[stream_indices]

    train_metadata = metadata.loc[train_indices]
    stream_metadata = metadata.loc[stream_indices]

    print("\n80/20 temporal split:")
    print("Training flows:", len(train_indices))
    print("Streaming flows:", len(stream_indices))

    # training phase
    training_hg = StreamingHypergraph()
    extractor = SClosenessFeatureExtractor()

    for source_ip, destination_ip, destination_port, timestamp in \
            train_hypergraph_data.itertuples(index=False):

        training_hg.add_flow(
            source_ip,
            destination_ip,
            destination_port,
            timestamp
        )

    s_values = extractor.calibrate(training_hg)

    print("\nCalibrated s-values:", s_values)

    training_centrality = extractor.compute_centrality(
        training_hg
    )

    training_feature_builder = HGIFeatureBuilder(
        extractor,
        ip_column="Destination IP"
    )

    X_train_hgi = training_feature_builder.build_features(
        train_stream_data,
        train_nrf,
        training_centrality
    )

    y_train = train_metadata["Label"].reset_index(drop=True)
    X_train_hgi = X_train_hgi.reset_index(drop=True)

    print("\nTraining HGI dataset:")
    print("X shape:", X_train_hgi.shape)
    print("y shape:", y_train.shape)

    print("\nTraining labels:")
    print(y_train.value_counts())

    model = train_hgi_model(
        X_train_hgi,
        y_train
    )

    # streaming phase
    hg = StreamingHypergraph()

    window_size = pd.Timedelta(minutes=5)
    time_window = TimeWindow(5)

    feature_builder = HGIFeatureBuilder(
        extractor,
        ip_column="Destination IP"
    )

    current_window_indices = []

    first_timestamp = stream_hypergraph_data[
        "Timestamp"
    ].iloc[0]

    time_window.start(first_timestamp)

    for index, row in stream_hypergraph_data.iterrows():

        source_ip = row["Source IP"]
        destination_ip = row["Destination IP"]
        destination_port = row["Destination Port"]
        timestamp = row["Timestamp"]

        hg.add_flow(
            source_ip,
            destination_ip,
            destination_port,
            timestamp
        )

        hg.remove_expired_flows(
            timestamp,
            window_size
        )

        current_window_indices.append(index)

        if time_window.is_complete(timestamp):

            print(
                "\nAnalysis window completed at:",
                timestamp
            )

            centrality = extractor.compute_centrality(hg)

            window_stream_data = stream_stream_data.loc[
                current_window_indices
            ]

            window_nrf = stream_nrf.loc[
                current_window_indices
            ]

            window_metadata = stream_metadata.loc[
                current_window_indices
            ]

            X_hgi = feature_builder.build_features(
                window_stream_data,
                window_nrf,
                centrality
            )

            y = window_metadata["Label"].reset_index(
                drop=True
            )

            X_hgi = X_hgi.reset_index(drop=True)

            print("\nHGI dataset:")
            print("X shape:", X_hgi.shape)
            print("y shape:", y.shape)

            predictions = model.predict(X_hgi)

            print("\nPrediction distribution:")
            print(
                pd.Series(predictions).value_counts()
            )

            print("\nActual distribution:")
            print(
                y.value_counts()
            )

            current_window_indices = []

            time_window.reset(timestamp)

    print("\nFinal streaming hypergraph:")
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