import pandas as pd
from dataManipulation import clean_dataset
from dataManipulation import prepare_data
from dataManipulation import prepare_labels
from dataManipulation import stream_flows
from hypergraph import StreamingHypergraph
csv_path = r"..\dataset\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
def main():
    print(f"LOading the csv file from, {csv_path}")
    raw_df = pd.read_csv(csv_path)
    print(f"Loaded {len(raw_df)} records.")

    # Dataset Cleaning
    print("-Cleaning Dataset-")
    clean_df = clean_dataset(raw_df)

    # Label changes to get only binary
    clean_df = prepare_labels(clean_df)
    
    # Feature Selection
    print("-Feature Selection-")
    stream_data, hypergraph_data, metadata,nrf = prepare_data(clean_df)

    # Streaming data (ippozhathek dataset ine stream kanakk akki ayakkuva)
    stream = stream_flows(stream_data)

    # HYPER GRAPH CREATION
    print ("-Hypergraph-")
    hg = StreamingHypergraph()

    flows = [
        ("10.0.0.1", "192.168.1.10", 80,  "10:00:00"),
        ("10.0.0.1", "192.168.1.10", 443, "10:01:00"),
        ("10.0.0.2", "192.168.1.10", 22,  "10:02:00"),
        ("10.0.0.1", "192.168.1.10", 80,  "10:03:00"),
        ("10.0.0.2", "192.168.1.10", 443, "10:04:00"),
    ]

    for source, destination, port, time in flows:
        hg.add_flow(
            source,
            destination,
            port,
            pd.Timestamp(f"2026-08-30 {time}")
        )

    print("Before expiration:")
    print(hg.get_hyperedges())
    
    window_size = pd.Timedelta(minutes=5)
    hg.remove_expired_flows(pd.Timestamp("2026-08-30 10:06:00"),window_size)
    print("\nAfter expiration:")
    print(hg.get_hyperedges())
    hg.remove_expired_flows(pd.Timestamp("2026-08-30 10:10:00"),window_size)
    print("\nAfter expiration 2:")
    print(hg.get_hyperedges())


if __name__ == "__main__":
    main()