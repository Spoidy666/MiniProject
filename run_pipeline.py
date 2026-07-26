import time
import pandas as pd
from hypergraph_nids.hypergraph import clean_dataset
from hypergraph_nids.simulation import NetworkSimulation

csv_path = r"c:\Users\vaish\Desktop\Projects\MiniProject\dataset\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

def main():
    print("==================================================")
    # 1. Load and clean the dataset
    print(f"Loading dataset from: {csv_path} ...")
    t0 = time.time()
    raw_df = pd.read_csv(csv_path)
    print(f"Loaded {len(raw_df)} records in {time.time() - t0:.2f} seconds.")
    
    print("Preprocessing and cleaning dataset...")
    t0 = time.time()
    clean_df = clean_dataset(raw_df)
    print(f"Cleaned dataset has {len(clean_df)} records. Preprocessing took {time.time() - t0:.2f} seconds.")
    print("Class distribution:")
    print(clean_df['Label'].value_counts())
    
    # 2. Setup the network simulation environment
    print("\nInitializing Network Simulation Environment...")
    t0 = time.time()
    # We use initial training size of 10000 records and batch size of 500
    sim = NetworkSimulation(clean_df, initial_train_size=10000, batch_size=500)
    print(f"Simulation environment initialized in {time.time() - t0:.2f} seconds.")
    
    # 3. Run the cases
    results = {}
    
    # Run each case for 15 epochs to demonstrate the trends while keeping execution quick.
    epochs = 15
    
    for case_id in [1, 2, 3, 4, 5, 6]:
        # Choose Threshold TH = 5 (representing threshold for retraining)
        # Note: Case 1 and 2 don't retrain, so TH is not used.
        t_case = time.time()
        history_df = sim.run_case(case_id=case_id, TH=5, epochs=epochs, num_computers=10)
        
        # Calculate summary metrics
        final_f1 = history_df['avg_f1_score'].iloc[-1]
        final_fnr = history_df['avg_f_negative_rate'].iloc[-1]
        mean_f1 = history_df['avg_f1_score'].mean()
        mean_fnr = history_df['avg_f_negative_rate'].mean()
        
        results[case_id] = {
            'final_f1': final_f1,
            'final_fnr': final_fnr,
            'mean_f1': mean_f1,
            'mean_fnr': mean_fnr,
            'history': history_df,
            'time_taken': time.time() - t_case
        }
        print(f"Case {case_id} completed in {time.time() - t_case:.2f} seconds. Final F1: {final_f1:.4f}, Final False Negative Rate: {final_fnr:.4f}")
        
    # 4. Print Summary Table
    print("\n=================== SIMULATION SUMMARY ===================")
    print(f"{'Case':<6} | {'Description':<40} | {'Mean F1':<10} | {'Final F1':<10} | {'Final FNR':<10} | {'Time (s)':<8}")
    print("-" * 95)
    
    descriptions = {
        1: "STATIC NIDS, 2 IP Pairs, No Adv",
        2: "STATIC NIDS, 16 IP Pairs, No Adv",
        3: "FTW Retraining, 16 IP Pairs, Yes Adv",
        4: "UALL Retraining, 16 IP Pairs, Yes Adv",
        5: "UALL Retraining, 16 IP Pairs, Yes Adv (All)",
        6: "UALL Retraining, 16 IP Pairs, Production"
    }
    
    for case_id, res in results.items():
        desc = descriptions[case_id]
        print(f"{case_id:<6} | {desc:<40} | {res['mean_f1']:<10.4f} | {res['final_f1']:<10.4f} | {res['final_fnr']:<10.4f} | {res['time_taken']:<8.2f}")
    print("==========================================================")

if __name__ == '__main__':
    main()
