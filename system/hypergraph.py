from collections import defaultdict, deque


class StreamingHypergraph:

    def __init__(self):
        self.hyperedges = defaultdict(lambda: defaultdict(int))
        self.flow_history = deque()

    def add_flow(self,source_ip,destination_ip,destination_port,timestamp):
    # Store the flow so that it can be removed later
        flow = (timestamp,source_ip,destination_ip,destination_port)
        self.flow_history.append(flow)
        self.hyperedges[source_ip][destination_port] += 1
        self.hyperedges[destination_ip][destination_port] += 1

    def get_hyperedge(self, ip):
        return dict(self.hyperedges.get(ip, {}))

    def get_hyperedges(self):
        return {
            ip: dict(ports)
            for ip, ports in self.hyperedges.items()
        }
    def remove_expired_flows(self, current_time, window_size):
        while self.flow_history:
            oldest_flow = self.flow_history[0]

            timestamp = oldest_flow[0]
            source_ip = oldest_flow[1]
            destination_ip = oldest_flow[2]
            destination_port = oldest_flow[3]

            # Stop once the oldest flow is still inside the window
            if current_time - timestamp <= window_size:
                break

            self.flow_history.popleft()

            self.hyperedges[source_ip][destination_port] -= 1

            if self.hyperedges[source_ip][destination_port] == 0:
                del self.hyperedges[source_ip][destination_port]

            self.hyperedges[destination_ip][destination_port] -= 1

            if self.hyperedges[destination_ip][destination_port] == 0:
                del self.hyperedges[destination_ip][destination_port]

            if not self.hyperedges[source_ip]:
                del self.hyperedges[source_ip]

            if not self.hyperedges[destination_ip]:
                del self.hyperedges[destination_ip]
    
    def number_of_hyperedges(self):
        return len(self.hyperedges)

    def number_of_vertices(self):
        vertices = set()

        for ports in self.hyperedges.values():
            vertices.update(ports)

        return len(vertices)