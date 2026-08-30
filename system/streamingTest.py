import pandas as pd

from hypergraph import StreamingHypergraph
from dataManipulation import clean_dataset, prepare_labels, prepare_data
from timeWindow import TimeWindow

CSV_PATH = r"..\dataset\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"


def main():

    dataset = pd.read_csv(CSV_PATH)

    df = clean_dataset(dataset)
    df = prepare_labels(df)

    stream_data, hypergraph_data, metadata, X_nrf = prepare_data(df)

    # -----------------------------
    # 2. Create our components
    # -----------------------------

    hg = StreamingHypergraph()

    window_size = pd.Timedelta(minutes=5) # Ith sliding window inte anu, old hyperedges remove cheyum aaa window inte end ill.

    time_window = TimeWindow(5) # Ith s closeness kandethanda time window anu

    # First timestamp in our stream
    first_timestamp = hypergraph_data["Timestamp"].iloc[0]

    time_window.start(first_timestamp)

    for source_ip, destination_ip, destination_port, timestamp in hypergraph_data.itertuples(index=False):

        # Update hypergraph
        hg.add_flow(
            source_ip,
            destination_ip,
            destination_port,
            timestamp
        )

        # Remove flows outside sliding window
        hg.remove_expired_flows(
            timestamp,
            window_size
        )

        # Check whether analysis window is complete
        if time_window.is_complete(timestamp):

            print(
                "Analysis window completed at:",
                timestamp
            )

            # s-closeness will eventually go here

            time_window.reset(timestamp)

    # -----------------------------
    # 4. Final state
    # -----------------------------

    print("\nFinal hypergraph:")
    print("Hyperedges:", hg.number_of_hyperedges())
    print("Vertices:", hg.number_of_vertices())

if __name__ == "__main__":
    main()