import pandas as pd
import numpy as np
import ipaddress

def is_valid_ip(value):
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False
def clean_dataset(dataset):

    print("\nOriginal shape before cleaning is :", dataset.shape,"\n") # (286467, 85) kitti
    df = dataset.copy()
    df.columns = [c.strip() for c in df.columns]
    nrf_cols = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Down/Up Ratio',
        'Protocol'
    ] # NRF = Network Raw Features
    required_cols = [
        'Source IP',
        'Destination IP',
        'Destination Port',
        'Timestamp',
        'Label'
    ] # This is used for the streaming system. 
    for col in nrf_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # To find the infinite values
    print("Removing nan, -inf, inf ")
    for col in nrf_cols:
        count = np.isinf(df[col]).sum()
        if count > 0:
            print(f"{col}: {count} infinite values")

    df[nrf_cols] = df[nrf_cols].replace([np.inf, -np.inf],np.nan)
    
    # missing_before = df[nrf_cols + required_cols].isna().sum() # To find why these flows will be dropped later? and drop avunna rows ill eth anu scene ann ariyan
    # print("\nMissing/invalid values before dropping:")
    # print(missing_before)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"],errors="coerce")
    df = df.dropna(subset=nrf_cols + required_cols)
    print(f"After removing -inf, inf and nan valued rows, the shape is {df.shape} \n")

    # Removing -flow rates
    before = len(df)

    df = df[df["Flow Duration"] >= 0]

    removed = before - len(df)

    print(f"Removed negative-duration flows: {removed} \n")


    print("After basic cleaning:", df.shape,"\n") # (286096, 85) Kitti


    # ip address okke valid ano enn checking
    print("-IP Address-")
    source_valid = df["Source IP"].apply(is_valid_ip)
    destination_valid = df["Destination IP"].apply(is_valid_ip)

    print("Invalid Source IPs:", (~source_valid).sum())
    print("Invalid Destination IPs:", (~destination_valid).sum(),"\n")

    # Ithinte answer source and destination inu 0 kittiyond onnum cheyunilla (ella ip um valid anu)

    # Port number correct ano alle enn checking
    print("-Ports-")
    print(df["Destination Port"].dtype)
    print("Ports below 0:", (df["Destination Port"] < 0).sum())
    print("Ports above 65535:",(df["Destination Port"] > 65535).sum())
    print("Missing destination ports:",df["Destination Port"].isna().sum(),"\n")

    # Ividem ellam valid vann athond changes onnum illa

    # Timestamps string object ill ninnn time ilott mattan
    print("-Timestamps-")
    print("Invalid timestamps:",df["Timestamp"].isna().sum())
    print("Earliest:", df["Timestamp"].min())
    print("Latest:", df["Timestamp"].max(),"\n")

    # Duplicates
    print("-Duplicates-")
    print("Duplicate rows:", df.duplicated().sum())
    duplicates = df[df.duplicated(keep=False)]
    print(duplicates)
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"Duplicate flows removed: {removed} \n")

    # SOrting
    print("-Sorting-")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print("First timestamp:",df["Timestamp"].iloc[0])
    print("\nLast timestamp:",df["Timestamp"].iloc[-1])
    print("Chronologically sorted:",df["Timestamp"].is_monotonic_increasing,"\n")
    print("After cleaning the final shape is : ", df.shape,"\n")

    return df


def prepare_labels(df):

    df = df.copy()

    df["Label"] = df["Label"].apply(
        lambda x: "BENIGN"
        if str(x).strip().upper() == "BENIGN"
        else "PORT SCAN"
    )
    df["Label"] = df["Label"].map({
            "BENIGN": 0,
            "PORT SCAN": 1
        })
    print(df["Label"].value_counts(),"\n")
    return df


def prepare_data(df):
    stream_cols = ["Flow ID","Source IP","Source Port","Destination IP","Destination Port","Protocol","Timestamp"] # Used for Flow/Stream information
    hypergraph_cols = ["Source IP","Destination IP","Destination Port","Timestamp"] # Used for hypergraph construction and other manipulations
    metadata_cols = ["Flow ID","Label"] # For Metadata
    nrf_cols = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Down/Up Ratio",
        "Protocol"
    ] # Ippozhathekk main paper inte same feature set edukkam pinna test cheythitt mattam


    X_nrf = df[nrf_cols].copy()
    stream_data = df[stream_cols].copy()
    hypergraph_data = df[hypergraph_cols].copy()
    metadata = df[metadata_cols].copy()
    print("Shape of X_nrf : ",X_nrf.shape)
    print("Shape of stream_data : ",stream_data.shape)
    print("Shape of hypergraph_data : ",hypergraph_data.shape)
    print("Shape of metadata : ",metadata.shape,"\n")
    return stream_data, hypergraph_data, metadata,X_nrf

def stream_flows(stream_data):

    for row in stream_data.itertuples(index=False):
        yield row
