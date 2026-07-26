import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.csgraph import connected_components, shortest_path

# Fixed weights for nonhacker pairs of IPs
NON_HACKER_WEIGHTS = np.array([0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.006, 0.005])

def clean_dataset(df):
    """
    Cleans the CIC-IDS2017 dataset by:
    1. Stripping whitespaces from column names.
    2. Filtering out rows with negative flow duration.
    3. Dropping rows with NaN/Inf values.
    4. Relabeling all non-BENIGN labels as 'PORT SCAN'.
    """
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]
    
    # Target columns for base network raw features (NRF)
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
    ]
    
    # Ensure they exist in dataframe
    for col in nrf_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            
    df_clean = df_clean.dropna(subset=nrf_cols + ['Source IP', 'Destination IP', 'Destination Port', 'Label'])
    df_clean = df_clean[df_clean['Flow Duration'] >= 0]
    
    # Relabel all attacks as PORT SCAN
    df_clean['Label'] = df_clean['Label'].apply(lambda x: 'BENIGN' if str(x).strip().upper() == 'BENIGN' else 'PORT SCAN')
    
    return df_clean

class HypergraphFeatureExtractor:
    def __init__(self):
        self.ips = []
        self.ports = []
        self.ip_to_idx = {}
        self.port_to_idx = {}
        self.M = 0
        self.k = 1
        self.s_values = []
        self.centralities = {}  # IP -> array of shape (11,)
        self.ip_ports_sizes = {} # IP -> number of ports
        self.flagged_pairs = set() # Set of flagged (src_ip, dst_ip)
        
    def fit(self, df):
        """
        Builds the hypergraph and computes s-closeness centralities for all IPs.
        """
        # Get unique IPs and ports
        self.ips = sorted(list(set(df['Source IP']).union(df['Destination IP'])))
        self.ports = sorted(list(df['Destination Port'].unique()))
        
        self.ip_to_idx = {ip: idx for idx, ip in enumerate(self.ips)}
        self.port_to_idx = {port: idx for idx, port in enumerate(self.ports)}
        
        # Build mapping IP -> set of ports
        ip_ports = {ip: set() for ip in self.ips}
        
        src_ip_idx = df.columns.get_loc('Source IP')
        dst_ip_idx = df.columns.get_loc('Destination IP')
        dst_port_idx = df.columns.get_loc('Destination Port')
        
        for row in df.values:
            src_ip = row[src_ip_idx]
            dst_ip = row[dst_ip_idx]
            dst_port = row[dst_port_idx]
            
            ip_ports[src_ip].add(dst_port)
            ip_ports[dst_ip].add(dst_port)
            
        self.ip_ports_sizes = {ip: len(ports) for ip, ports in ip_ports.items()}
        
        # Build sparse matrix A
        row_indices = []
        col_indices = []
        for ip, p_set in ip_ports.items():
            i_idx = self.ip_to_idx[ip]
            for p in p_set:
                row_indices.append(i_idx)
                col_indices.append(self.port_to_idx[p])
                
        data = np.ones(len(row_indices), dtype=np.int32)
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(len(self.ips), len(self.ports)))
        
        # M is the max hyperedge size
        hyperedge_sizes = np.array(A.sum(axis=1)).flatten()
        self.M = int(np.max(hyperedge_sizes)) if len(hyperedge_sizes) > 0 else 0
        
        # skip-interval k
        self.k = max(1, int(round((0.7 * self.M - 3) / 10)))
        self.s_values = [3 + n * self.k for n in range(11)]
        
        # Compute A * A^T
        C = A.dot(A.T)
        
        # Compute centralities for each s
        centralities_matrix = np.zeros((len(self.ips), 11))
        
        for n_idx, s in enumerate(self.s_values):
            C_upper = sp.triu(C, k=1)
            C_coo = C_upper.tocoo()
            mask = C_coo.data >= s
            
            edges = list(zip(C_coo.row[mask], C_coo.col[mask]))
            
            G = nx.Graph()
            G.add_nodes_from(range(len(self.ips)))
            G.add_edges_from(edges)
            
            # Find connected components
            n_components, labels = connected_components(csgraph=nx.to_scipy_sparse_array(G), directed=False, return_labels=True)
            
            comp_nodes = [[] for _ in range(n_components)]
            for node, label in enumerate(labels):
                comp_nodes[label].append(node)
                
            for label in range(n_components):
                nodes = comp_nodes[label]
                size = len(nodes)
                if size <= 1:
                    continue
                elif size == 2:
                    centralities_matrix[nodes[0], n_idx] = 1.0
                    centralities_matrix[nodes[1], n_idx] = 1.0
                else:
                    # Run shortest path for this component
                    sub_adj = nx.to_scipy_sparse_array(G)[nodes, :][:, nodes]
                    dist_matrix = shortest_path(csgraph=sub_adj, directed=False, unweighted=True)
                    dist_matrix[np.isinf(dist_matrix)] = 0
                    sum_dists = dist_matrix.sum(axis=1)
                    for idx, node in enumerate(nodes):
                        sum_dist = sum_dists[idx]
                        if sum_dist > 0:
                            centralities_matrix[node, n_idx] = (size - 1) / sum_dist
                            
        # Store in dict
        for idx, ip in enumerate(self.ips):
            self.centralities[ip] = centralities_matrix[idx]
            
        # Re-run behavioral analytics to populate flagged pairs
        self.detect_port_scans(df)
            
    def detect_port_scans(self, df):
        """
        Algorithm 1: Process to Detect Potential Port Scan Activities
        """
        src_ip_idx = df.columns.get_loc('Source IP')
        dst_ip_idx = df.columns.get_loc('Destination IP')
        
        for row in df.values:
            src_ip = row[src_ip_idx]
            dst_ip = row[dst_ip_idx]
            
            if (src_ip, dst_ip) in self.flagged_pairs:
                continue
                
            dst_centrality = self.centralities.get(dst_ip, np.zeros(11))
            last_six = dst_centrality[5:]
            converted_last_six = np.where(last_six >= 0.95, 1, 0)
            
            if np.sum(converted_last_six) >= 2:
                self.flagged_pairs.add((src_ip, dst_ip))
                
    def extract_features(self, df, use_labels=True):
        """
        Extracts NRF, HGI, and HGA feature sets for the records in df.
        """
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
        ]
        
        X_nrf = df[nrf_cols].values.astype(np.float32)
        n_records = len(df)
        
        X_cc = np.zeros((n_records, 11), dtype=np.float32)
        X_es_ip = np.zeros(n_records, dtype=np.float32)
        X_ed_ip = np.zeros(n_records, dtype=np.float32)
        
        src_ip_idx = df.columns.get_loc('Source IP')
        dst_ip_idx = df.columns.get_loc('Destination IP')
        
        if use_labels:
            labels = df['Label'].values
            is_benign = (labels == 'BENIGN')
        else:
            is_benign = np.zeros(n_records, dtype=bool)
            for idx, row in enumerate(df.values):
                src_ip = row[src_ip_idx]
                dst_ip = row[dst_ip_idx]
                if (src_ip, dst_ip) not in self.flagged_pairs:
                    is_benign[idx] = True
                    
        for idx, row in enumerate(df.values):
            src_ip = row[src_ip_idx]
            dst_ip = row[dst_ip_idx]
            
            if is_benign[idx]:
                X_cc[idx] = NON_HACKER_WEIGHTS
            else:
                X_cc[idx] = self.centralities.get(dst_ip, np.zeros(11))
                
            X_es_ip[idx] = self.ip_ports_sizes.get(src_ip, 1)
            X_ed_ip[idx] = self.ip_ports_sizes.get(dst_ip, 1)
            
        sum_cc = X_cc.sum(axis=1, keepdims=True)
        X_hgi = np.hstack([X_nrf, X_cc, sum_cc])
        
        last_cc = X_cc[:, 10:11]
        sum_sizes = (X_es_ip + X_ed_ip).reshape(-1, 1)
        X_hga = np.hstack([
            X_nrf,
            last_cc,
            sum_cc,
            X_es_ip.reshape(-1, 1),
            X_ed_ip.reshape(-1, 1),
            sum_sizes
        ])
        
        return X_nrf, X_hgi, X_hga
