from datetime import datetime, timedelta

import hypergraph
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# IMPORT PRODUCTION FEATURE EXTRACTOR INSTEAD OF COMMENTED-OUT S_CLOSENESS
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
from sClosenessFeatureExtractor import SClosenessFeatureExtractor
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
hg = hypergraph.StreamingHypergraph()


base_time = datetime.now()

# --------------------------------------------------
# IP 1 - accesses many common ports
# --------------------------------------------------

ports_ip1 = [21, 22, 23, 25, 53, 80, 110, 135, 139, 443]

for i, port in enumerate(ports_ip1):
    hg.add_flow(
        "192.168.1.10",
        "10.0.0.1",
        port,
        base_time + timedelta(seconds=i)
    )


# --------------------------------------------------
# IP 2 - accesses a small number of common ports
# --------------------------------------------------

ports_ip2 = [80, 443]

for i, port in enumerate(ports_ip2):
    hg.add_flow(
        "192.168.1.20",
        "10.0.0.1",
        port,
        base_time + timedelta(seconds=20 + i)
    )


# --------------------------------------------------
# IP 3 - accesses completely different ports
# --------------------------------------------------

ports_ip3 = [8080, 8443, 9000, 9999]

for i, port in enumerate(ports_ip3):
    hg.add_flow(
        "192.168.1.30",
        "10.0.0.2",
        port,
        base_time + timedelta(seconds=30 + i)
    )


print("Current Hypergraph:")
print(hg.get_hyperedges())

print("\nHyperedge sizes:")

for ip, ports in hg.get_hyperedges().items():
    print(ip, len(ports))

# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# CALIBRATE AND COMPUTE CENTRALITY USING PRODUCTION EXTRACTOR
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
extractor = SClosenessFeatureExtractor()
s_values = extractor.calibrate(hg)
print(f"\nCalibrated s-values (count={len(s_values)}):", s_values)

results = extractor.compute_centrality(hg)

print("\nS-closeness Results:")
for s, scores in results.items():
    print(f"  s={s}: {scores}")
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000