import os
import time
import json
import pandas as pd
import numpy as np

# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# METRIC AND PERSISTENCE IMPORTS
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000

from hypergraph import StreamingHypergraph
from dataManipulation import (
    clean_dataset,
    prepare_labels,
    prepare_data
)
from timeWindow import TimeWindow
from sClosenessFeatureExtractor import SClosenessFeatureExtractor
from hgiFeatureBuilder import HGIFeatureBuilder

# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# MODEL TRAINING AND ARTIFACT LOADING FUNCTIONS
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
from hgi_model import train_hgi_model, load_hgi_model
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000

# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# ROBUST PATH CONFIGURATION FOR ARTIFACTS AND OUTPUTS
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "dataset", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models")
MODEL_FILE = os.path.join(MODEL_DIR, "hgi_lightgbm_model.pkl")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "streaming_window_metrics.csv")
SUMMARY_JSON = os.path.join(SCRIPT_DIR, "streaming_summary.json")
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000


def main():
    print("=" * 80)
    print(" ADAPTIVE STREAMING HYPERGRAPH INTRUSION DETECTION SYSTEM (NIDS)")
    print(" Target: Port Scan Detection | Dataset: CIC-IDS-2017")
    print("=" * 80)

    # 1. Load and prepare dataset
    print(f"\n[1/5] Loading raw network flows from: {CSV_PATH}")
    t0 = time.time()
    dataset = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(dataset):,} raw records in {time.time() - t0:.2f}s")

    print("\n[2/5] Cleaning dataset and formatting binary labels...")
    t0 = time.time()
    df = clean_dataset(dataset)
    df = prepare_labels(df)
    stream_data, hypergraph_data, metadata, X_nrf = prepare_data(df)
    print(f"Preprocessed {len(df):,} valid flows in {time.time() - t0:.2f}s")

    # 2. 80/20 Temporal Split
    print("\n[3/5] Performing Chronological 80/20 Train/Stream Split...")
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

    print(f"  Training Flows (First 80%) : {len(train_indices):,}")
    print(f"  Streaming Flows (Held-out 20%): {len(stream_indices):,}")

    # 3. Training Phase (80% Split)
    print("\n[4/5] Training Phase: Constructing Hypergraph & Calibrating Model...")
    t0_train = time.time()
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

    print(f"  Training Graph built: {training_hg.number_of_hyperedges():,} Hyperedges, {training_hg.number_of_vertices():,} Vertices")

    s_values = extractor.calibrate(training_hg)
    print(f"  Calibrated 11 s-values: {s_values}")

    print("  Computing training hypergraph centrality...")
    t0_cent = time.time()
    training_centrality = extractor.compute_centrality(training_hg)
    print(f"  Training centrality computed in {time.time() - t0_cent:.2f}s")

    training_feature_builder = HGIFeatureBuilder(
        extractor,
        ip_column="Destination IP"
    )

    print("  Building 21-dimensional HGI features for training flows...")
    t0_feat = time.time()
    X_train_hgi = training_feature_builder.build_features(
        train_stream_data,
        train_nrf,
        training_centrality
    )
    print(f"  HGI features generated in {time.time() - t0_feat:.2f}s (Shape: {X_train_hgi.shape})")

    y_train = train_metadata["Label"].reset_index(drop=True)
    X_train_hgi = X_train_hgi.reset_index(drop=True)

    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    # TRAIN AND SAVE MODEL ARTIFACT ON 80% TRAIN DATASET
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    feature_names = training_feature_builder.get_feature_names()
    print(f"\n  Training LightGBM model and saving artifact to: {MODEL_FILE}")
    _ = train_hgi_model(
        X=X_train_hgi,
        y=y_train,
        save_path=MODEL_FILE,
        s_values=s_values,
        feature_names=feature_names
    )
    print(f"  Training phase completed in {time.time() - t0_train:.2f}s")

    # Load the saved model artifact to ensure persistence and use loaded model for streaming
    artifact = load_hgi_model(MODEL_FILE)
    streaming_model = artifact["model"]
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

    # 4. Streaming Phase (Held-out 20% Split)
    print("\n[5/5] Streaming Phase: Simulating 5-Minute Sliding-Window Traffic...")
    hg = StreamingHypergraph()
    window_size = pd.Timedelta(minutes=5)
    time_window = TimeWindow(5)

    feature_builder = HGIFeatureBuilder(
        extractor,
        ip_column="Destination IP"
    )

    current_window_indices = []
    first_timestamp = stream_hypergraph_data["Timestamp"].iloc[0]
    time_window.start(first_timestamp)

    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    # STREAMING METRICS LOGGERS AND ACCUMULATORS
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    window_records = []
    window_count = 0
    total_streaming_flows_processed = 0

    cumulative_tp = 0
    cumulative_fp = 0
    cumulative_tn = 0
    cumulative_fn = 0
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

    t_stream_start = time.time()

    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    # HELPER FUNCTION FOR RUNNING WINDOW ANALYSIS AND TESTING METRICS
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    def process_analysis_window(w_indices, w_timestamp, w_num):
        nonlocal cumulative_tp, cumulative_fp, cumulative_tn, cumulative_fn, total_streaming_flows_processed

        if not w_indices:
            return

        t_w_start = time.time()
        w_flows_count = len(w_indices)
        total_streaming_flows_processed += w_flows_count

        # Hypergraph state inspection
        active_hyperedges = hg.number_of_hyperedges()
        active_vertices = hg.number_of_vertices()
        max_edge_size = extractor.get_max_hyperedge_size(hg)

        # Compute centrality for current sliding hypergraph
        t_w_cent = time.time()
        centrality = extractor.compute_centrality(hg)
        cent_duration = time.time() - t_w_cent

        # Prepare window feature slices
        w_stream_data = stream_stream_data.loc[w_indices]
        w_nrf = stream_nrf.loc[w_indices]
        w_metadata = stream_metadata.loc[w_indices]

        # Build HGI features
        t_w_feat = time.time()
        X_hgi = feature_builder.build_features(
            w_stream_data,
            w_nrf,
            centrality
        )
        feat_duration = time.time() - t_w_feat

        # Measure s-value zero ratio (Diagnose s-value mismatch in streaming windows)
        s_cols = [f"s_closeness_{s}" for s in extractor.s_values]
        s_features_matrix = X_hgi[s_cols].values
        zero_s_count = np.sum(s_features_matrix == 0.0)
        total_s_entries = s_features_matrix.size
        zero_s_ratio = (zero_s_count / total_s_entries * 100.0) if total_s_entries > 0 else 0.0

        y_true = w_metadata["Label"].values
        X_hgi = X_hgi.reset_index(drop=True)

        # Run model prediction using saved artifact model
        t_w_pred = time.time()
        y_pred = streaming_model.predict(X_hgi)
        pred_duration = time.time() - t_w_pred

        total_w_duration = time.time() - t_w_start

        # Calculate Testing Metrics for this window
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        w_acc = float(accuracy_score(y_true, y_pred))
        w_prec = float(precision_score(y_true, y_pred, zero_division=0))
        w_rec = float(recall_score(y_true, y_pred, zero_division=0))
        w_f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Accumulate global metrics
        cumulative_tp += tp
        cumulative_fp += fp
        cumulative_tn += tn
        cumulative_fn += fn

        # Log metrics record
        record = {
            "window_id": w_num,
            "timestamp": str(w_timestamp),
            "flow_count": int(w_flows_count),
            "active_hyperedges": int(active_hyperedges),
            "active_vertices": int(active_vertices),
            "max_hyperedge_size": int(max_edge_size),
            "s_closeness_zero_ratio_pct": round(zero_s_ratio, 2),
            "accuracy": round(w_acc, 6),
            "precision": round(w_prec, 6),
            "recall": round(w_rec, 6),
            "f1_score": round(w_f1, 6),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "processing_time_sec": round(total_w_duration, 4),
            "centrality_time_sec": round(cent_duration, 4),
            "feature_time_sec": round(feat_duration, 4),
            "predict_time_sec": round(pred_duration, 4)
        }
        window_records.append(record)

        # Print testing metrics for this window
        print("-" * 80)
        print(f"[Window #{w_num:02d} Evaluated] Timestamp: {w_timestamp}")
        print(f"  Traffic Stats   : {w_flows_count:,} Flows | {active_hyperedges} Hyperedges | {active_vertices} Vertices")
        print(f"  Graph Topology  : Max Edge Size: {max_edge_size} | s-Closeness Zero Ratio: {zero_s_ratio:.1f}%")
        print(f"  Window Metrics  : Acc: {w_acc*100:.3f}% | F1: {w_f1:.4f} | Prec: {w_prec:.4f} | Rec: {w_rec:.4f}")
        print(f"  Confusion Matrix: TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
        print(f"  Computation Time: Total: {total_w_duration:.3f}s (Centrality: {cent_duration:.3f}s, Features: {feat_duration:.3f}s)")
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    # FAST STREAMING LOOP OVER HELD-OUT FLOWS
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    for row in stream_hypergraph_data.itertuples():
        index = row.Index
        source_ip = row._1
        destination_ip = row._2
        destination_port = row._3
        timestamp = row._4

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
            window_count += 1
            process_analysis_window(
                w_indices=current_window_indices,
                w_timestamp=timestamp,
                w_num=window_count
            )
            current_window_indices = []
            time_window.reset(timestamp)

    # Process any remaining flows at stream conclusion
    if current_window_indices:
        window_count += 1
        last_timestamp = stream_hypergraph_data["Timestamp"].iloc[-1]
        process_analysis_window(
            w_indices=current_window_indices,
            w_timestamp=last_timestamp,
            w_num=window_count
        )
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

    total_stream_duration = time.time() - t_stream_start

    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    # CUMULATIVE STREAMING TESTING METRICS AGGREGATION & REPORTING
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    total_evaluated = cumulative_tp + cumulative_fp + cumulative_tn + cumulative_fn

    overall_acc = (cumulative_tp + cumulative_tn) / total_evaluated if total_evaluated > 0 else 0.0
    overall_prec = cumulative_tp / (cumulative_tp + cumulative_fp) if (cumulative_tp + cumulative_fp) > 0 else 0.0
    overall_rec = cumulative_tp / (cumulative_tp + cumulative_fn) if (cumulative_tp + cumulative_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0
    overall_fpr = cumulative_fp / (cumulative_fp + cumulative_tn) if (cumulative_fp + cumulative_tn) > 0 else 0.0
    overall_fnr = cumulative_fn / (cumulative_fn + cumulative_tp) if (cumulative_fn + cumulative_tp) > 0 else 0.0

    mean_window_time = float(np.mean([w["processing_time_sec"] for w in window_records])) if window_records else 0.0
    mean_zero_s_ratio = float(np.mean([w["s_closeness_zero_ratio_pct"] for w in window_records])) if window_records else 0.0

    print("\n" + "=" * 80)
    print(" CUMULATIVE STREAMING EVALUATION RESULTS (HELD-OUT 20% STREAM)")
    print("=" * 80)
    print(f"  Total Streaming Flows Processed: {total_evaluated:,}")
    print(f"  Total 5-Minute Analysis Windows: {window_count}")
    print(f"  Total Streaming Elapsed Time   : {total_stream_duration:.2f}s (Avg {mean_window_time:.3f}s / window)")
    print(f"  Average s-Closeness Zero Ratio : {mean_zero_s_ratio:.2f}% (s-value mismatch indicator)")
    print("-" * 80)
    print(f"  Overall Streaming Accuracy     : {overall_acc * 100:.4f}%")
    print(f"  Overall Streaming Precision    : {overall_prec:.6f}")
    print(f"  Overall Streaming Recall (DR)  : {overall_rec:.6f}")
    print(f"  Overall Streaming F1-Score     : {overall_f1:.6f}")
    print(f"  False Positive Rate (FPR)      : {overall_fpr * 100:.4f}%")
    print(f"  False Negative Rate (FNR)      : {overall_fnr * 100:.4f}%")
    print("-" * 80)
    print("  Cumulative Confusion Matrix:")
    print(f"    True Positives  (Attack correctly detected) : {cumulative_tp:,}")
    print(f"    True Negatives  (Benign correctly allowed)  : {cumulative_tn:,}")
    print(f"    False Positives (False Alarm on Benign)     : {cumulative_fp:,}")
    print(f"    False Negatives (Missed PortScan Attack)    : {cumulative_fn:,}")
    print("=" * 80)

    # Export window-by-window metrics to CSV
    metrics_df = pd.DataFrame(window_records)
    metrics_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[Exported] Detailed window metrics saved to: {RESULTS_CSV}")

    # Export cumulative summary to JSON
    summary_data = {
        "dataset": CSV_PATH,
        "total_streaming_flows": total_evaluated,
        "total_windows": window_count,
        "total_streaming_time_sec": round(total_stream_duration, 2),
        "mean_window_processing_time_sec": round(mean_window_time, 4),
        "mean_s_closeness_zero_ratio_pct": round(mean_zero_s_ratio, 2),
        "overall_accuracy": round(overall_acc, 6),
        "overall_precision": round(overall_prec, 6),
        "overall_recall": round(overall_rec, 6),
        "overall_f1_score": round(overall_f1, 6),
        "false_positive_rate": round(overall_fpr, 6),
        "false_negative_rate": round(overall_fnr, 6),
        "confusion_matrix": {
            "true_positives": cumulative_tp,
            "false_positives": cumulative_fp,
            "true_negatives": cumulative_tn,
            "false_negatives": cumulative_fn
        },
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[Exported] Cumulative summary metrics saved to: {SUMMARY_JSON}\n")
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000


if __name__ == "__main__":
    main()