import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .models import EnsembleNIDS
from .hypergraph import HypergraphFeatureExtractor, clean_dataset

def generate_adversarial_example_proxy(substitute_model, x, features_to_perturb=[0, 1, 3, 5, 6, 7], step_size=0.1, max_steps=10):
    """
    Fast proxy for ZOO attack. Perturbs raw features of x (shape: (9,)) to decrease
    the probability of being classified as PORT SCAN (class 1) by the substitute model.
    """
    x_adv = x.copy()
    p_orig = substitute_model.predict_proba(x_adv.reshape(1, -1))[0, 1]
    if p_orig < 0.5:
        return x_adv, True
        
    for _ in range(max_steps):
        proposal = x_adv.copy()
        idx = np.random.choice(features_to_perturb)
        proposal[idx] += np.random.normal(0, step_size * (x_adv[idx] + 1e-5))
        if idx in [0, 1, 2, 3, 4, 5, 6]:
            proposal[idx] = max(0, proposal[idx])
            
        p_new = substitute_model.predict_proba(proposal.reshape(1, -1))[0, 1]
        if p_new < p_orig:
            x_adv = proposal
            p_orig = p_new
            if p_orig < 0.5:
                return x_adv, True
                
    return x_adv, p_orig >= 0.55

class NetworkSimulation:
    def __init__(self, clean_df, initial_train_size=10000, batch_size=500):
        # Shuffle the dataset to get a balanced initial training set
        self.df = clean_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        self.initial_train_size = initial_train_size
        self.batch_size = batch_size
        
        # Split into initial training set and simulation set
        self.train_df = self.df.iloc[:self.initial_train_size].copy()
        self.sim_df = self.df.iloc[self.initial_train_size:].copy()
        
        # Build initial Hypergraph on training set
        self.hg_extractor = HypergraphFeatureExtractor()
        self.hg_extractor.fit(self.train_df)
        
        # Generate initial features
        self.X_nrf_train, self.X_hgi_train, self.X_hga_train = self.hg_extractor.extract_features(self.train_df)
        self.y_train = (self.train_df['Label'] == 'PORT SCAN').astype(int).values
        
        # Train initial Ensemble NIDS
        self.ensemble = EnsembleNIDS()
        self.ensemble.fit(self.X_nrf_train, self.X_hgi_train, self.X_hga_train, self.y_train)
        
        # Fit substitute model for ZOO attack
        from lightgbm import LGBMClassifier
        self.substitute_model = LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)
        self.substitute_model.fit(self.X_nrf_train, self.y_train)
        
    def run_case(self, case_id, TH=5, epochs=30, num_computers=10):
        """
        Runs the simulation for a given Case ID:
        1: STATIC NIDS, 2 IP Pairs, No Adversarial Examples.
        2: STATIC NIDS, 16 IP Pairs, No Adversarial Examples.
        3: FTW Update Rule, 16 IP Pairs, Yes Adversarial Examples, Threshold TH.
        4: UALL Update Rule, 16 IP Pairs, Yes Adversarial Examples, Threshold TH.
        5: UALL Update Rule, 16 IP Pairs, Yes Adversarial Examples, Threshold TH, All Attacks.
        6: UALL Update Rule, 16 IP Pairs, Yes Adversarial Examples, Threshold TH, Production (Behavioral Analytics).
        """
        print(f"\n--- Starting Case {case_id} (TH={TH}, Epochs={epochs}) ---")
        
        # Reset hypergraph extractor flagged pairs
        self.hg_extractor.flagged_pairs = set()
        self.hg_extractor.detect_port_scans(self.train_df)
        
        # Determine number of IP Pairs
        ip_pairs_count = 2 if case_id == 1 else 16
        
        # Get top source IPs of port scans in the simulation set
        scan_df = self.sim_df[self.sim_df['Label'] == 'PORT SCAN']
        top_scanners = scan_df['Source IP'].value_counts().index[:ip_pairs_count].tolist()
        
        # Filter simulation traffic for the chosen scanners (plus benign traffic)
        case_sim_df = self.sim_df[self.sim_df['Source IP'].isin(top_scanners) | (self.sim_df['Label'] == 'BENIGN')].copy()
        
        # Initialize computer agents
        computers = []
        for i in range(num_computers):
            computers.append({
                'ensemble': self.ensemble.copy(),
                'infiltrated_count': 0,
                'evaded_attacks': [],
                'balanced_benign': []
            })
            
        history = []
        
        # Create batches of traffic
        benign_pool = case_sim_df[case_sim_df['Label'] == 'BENIGN']
        attack_pool = case_sim_df[case_sim_df['Label'] == 'PORT SCAN']
        
        n_benign_per_batch = int(self.batch_size * 0.75)
        n_attack_per_batch = int(self.batch_size * 0.25)
        
        print(f"Pool size - Benign: {len(benign_pool)}, Attack: {len(attack_pool)}")
        
        benign_indices = np.random.choice(len(benign_pool), size=n_benign_per_batch * num_computers * epochs, replace=True)
        attack_indices = np.random.choice(len(attack_pool), size=n_attack_per_batch * num_computers * epochs, replace=True)
        
        batch_idx = 0
        
        for epoch in range(epochs):
            epoch_fn_rates = []
            epoch_f1_scores = []
            
            for comp_idx in range(num_computers):
                comp = computers[comp_idx]
                
                b_b_indices = benign_indices[batch_idx * n_benign_per_batch : (batch_idx + 1) * n_benign_per_batch]
                b_a_indices = attack_indices[batch_idx * n_attack_per_batch : (batch_idx + 1) * n_attack_per_batch]
                batch_idx += 1
                
                batch_benign = benign_pool.iloc[b_b_indices].copy()
                batch_attack = attack_pool.iloc[b_a_indices].copy()
                
                # Apply adversarial examples if enabled (Cases 3-6)
                if case_id in [3, 4, 5, 6]:
                    X_attack_raw = batch_attack[[
                        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                        'Flow Bytes/s', 'Flow Packets/s', 'Down/Up Ratio', 'Protocol'
                    ]].values
                    
                    perturbed_attacks = []
                    for x in X_attack_raw:
                        x_adv, is_adv = generate_adversarial_example_proxy(self.substitute_model, x)
                        perturbed_attacks.append(x_adv)
                        
                    batch_attack[[
                        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                        'Flow Bytes/s', 'Flow Packets/s', 'Down/Up Ratio', 'Protocol'
                    ]] = perturbed_attacks
                    
                # Combine benign and attack into a single batch
                batch = pd.concat([batch_benign, batch_attack]).sample(frac=1.0).reset_index(drop=True)
                
                # Extract features for prediction
                use_labels = False if case_id == 6 else True
                X_nrf, X_hgi, X_hga = self.hg_extractor.extract_features(batch, use_labels=use_labels)
                y_true = (batch['Label'] == 'PORT SCAN').astype(int).values
                
                # Classify batch using computer's active ensemble NIDS
                y_pred = comp['ensemble'].predict(X_nrf, X_hgi, X_hga)
                
                # Metrics
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                fn_mask = (y_true == 1) & (y_pred == 0)
                fn_count = np.sum(fn_mask)
                fn_rate = fn_count / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
                
                epoch_fn_rates.append(fn_rate)
                epoch_f1_scores.append(f1)
                
                # Handle retraining trigger (if not STATIC)
                if case_id in [3, 4, 5, 6]:
                    if case_id == 6:
                        # Build hypergraph on current batch to flag scanners
                        batch_hg = HypergraphFeatureExtractor()
                        batch_hg.fit(batch)
                        
                        # Add flagged pairs to global set
                        self.hg_extractor.flagged_pairs.update(batch_hg.flagged_pairs)
                        
                        src_ip_idx = batch.columns.get_loc('Source IP')
                        dst_ip_idx = batch.columns.get_loc('Destination IP')
                        
                        for idx, row in enumerate(batch.values):
                            src_ip = row[src_ip_idx]
                            dst_ip = row[dst_ip_idx]
                            if (src_ip, dst_ip) in self.hg_extractor.flagged_pairs and y_pred[idx] == 0:
                                comp['evaded_attacks'].append(batch.iloc[idx])
                                rand_benign = batch_benign.iloc[np.random.choice(len(batch_benign))]
                                comp['balanced_benign'].append(rand_benign)
                                comp['infiltrated_count'] += 1
                    else:
                        if fn_count > 0:
                            evaded_records = batch[fn_mask]
                            for _, r in evaded_records.iterrows():
                                comp['evaded_attacks'].append(r)
                                rand_benign = batch_benign.iloc[np.random.choice(len(batch_benign))]
                                comp['balanced_benign'].append(rand_benign)
                                
                            comp['infiltrated_count'] += fn_count
                            
                    # Retrain if threshold exceeded
                    if comp['infiltrated_count'] >= TH:
                        retrain_df = pd.DataFrame(comp['evaded_attacks'] + comp['balanced_benign'])
                        retrain_df = pd.concat([self.train_df, retrain_df]).drop_duplicates().reset_index(drop=True)
                        
                        self.hg_extractor.fit(retrain_df)
                        
                        X_nrf_retrain, X_hgi_retrain, X_hga_retrain = self.hg_extractor.extract_features(retrain_df)
                        y_retrain = (retrain_df['Label'] == 'PORT SCAN').astype(int).values
                        
                        if case_id in [4, 5, 6]:
                            # UALL: Retrain and update all models
                            new_ensemble = EnsembleNIDS()
                            new_ensemble.fit(X_nrf_retrain, X_hgi_retrain, X_hga_retrain, y_retrain)
                            
                            for c in computers:
                                c['ensemble'] = new_ensemble.copy()
                                c['infiltrated_count'] = 0
                                c['evaded_attacks'] = []
                                c['balanced_benign'] = []
                                
                        elif case_id == 3:
                            # FTW: Retrain candidate model (e.g. HGI)
                            from lightgbm import LGBMClassifier
                            candidate_hgi = LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)
                            candidate_hgi.fit(X_hgi_retrain, y_retrain)
                            
                            # Evaluate on training data
                            feature_map_train = {
                                'nrf': self.X_nrf_train,
                                'hgi': self.X_hgi_train,
                                'hga': self.X_hga_train
                            }
                            
                            f1_scores = []
                            for model, feat_type in comp['ensemble'].models_and_feats:
                                pred = model.predict(feature_map_train[feat_type])
                                f1_scores.append(f1_score(self.y_train, pred, zero_division=0))
                                
                            worst_idx = np.argmin(f1_scores)
                            
                            X_hgi_train_updated = self.hg_extractor.extract_features(self.train_df)[1]
                            pred_candidate = candidate_hgi.predict(X_hgi_train_updated)
                            f1_candidate = f1_score(self.y_train, pred_candidate, zero_division=0)
                            
                            if f1_candidate > f1_scores[worst_idx]:
                                comp['ensemble'].models_and_feats[worst_idx] = (candidate_hgi, 'hgi')
                                    
                            comp['infiltrated_count'] = 0
                            comp['evaded_attacks'] = []
                            comp['balanced_benign'] = []
                            
            avg_fn_rate = np.mean(epoch_fn_rates)
            avg_f1_score = np.mean(epoch_f1_scores)
            history.append({
                'epoch': epoch + 1,
                'avg_f_negative_rate': avg_fn_rate,
                'avg_f1_score': avg_f1_score
            })
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Avg F1 Score: {avg_f1_score:.4f} | Avg False Negative Rate: {avg_fn_rate:.4f}")
                
        return pd.DataFrame(history)
