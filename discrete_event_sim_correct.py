
Discrete-event simulation for smart grid communication (G/D/1 queue, strict priority)
Validates Network Calculus PBOO bound. Author: Koroma Samuel Ansumana

import heapq
import numpy as np

# ========== Parameters ==========
LINK_RATE_BPS = 10_000_000          # 10 Mbps bottleneck
PROP_DELAY_S = 0.017                # 17 ms total propagation+processing

# Critical flow (fault messages)
CRIT_BURST_BYTES = 128_000          # 500 packets * 256 bytes
CRIT_PACKET_SIZE = 256
PERIODIC_RATE_BPS = 25_600          # 100 packets/s * 256 bytes

# Background flow (cross-traffic)
BG_MEAN_RATE_BPS = 2_000_000        # 2 Mbps
BG_PACKET_SIZE = 512
BG_BURST_PROB = 0.1
BG_BURST_EXTRA = 5000               # extra bytes per burst

SIM_DURATION = 60                   # seconds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========== Event generation ==========
def generate_events():
    events = []  # (time, flow_type, packet_id, size_bytes)
    pid = 0

    # Critical: periodic monitoring
    t = 0
    period = CRIT_PACKET_SIZE / PERIODIC_RATE_BPS
    while t < SIM_DURATION:
        events.append((t, 'crit', pid, CRIT_PACKET_SIZE))
        pid += 1
        t += period

    # Critical: fault burst at t=10s
    burst_packets = int(CRIT_BURST_BYTES / CRIT_PACKET_SIZE)
    for i in range(burst_packets):
        events.append((10.0 + i * 0.0001, 'crit', pid, CRIT_PACKET_SIZE))
        pid += 1

    # Background: Poisson + random bursts
    t = 0
    avg_inter = BG_PACKET_SIZE / BG_MEAN_RATE_BPS
    while t < SIM_DURATION:
        events.append((t, 'bg', pid, BG_PACKET_SIZE))
        pid += 1
        if np.random.random() < BG_BURST_PROB:
            burst_len = np.random.poisson(BG_BURST_EXTRA / BG_PACKET_SIZE)
            for i in range(burst_len):
                events.append((t + i * 0.0002, 'bg', pid, BG_PACKET_SIZE))
                pid += 1
        t += np.random.exponential(avg_inter)

    events.sort(key=lambda x: x[0])
    return events

class Packet:
    __slots__ = ('arr_time', 'size', 'flow_type', 'pid')
    def __init__(self, arr_time, size, flow_type, pid):
        self.arr_time = arr_time
        self.size = size
        self.flow_type = flow_type
        self.pid = pid

# ========== Main simulation ==========
def run_simulation():
    events = generate_events()
    crit_queue = []   # (arr_time, pid, packet)
    bg_queue = []
    server_free_time = 0.0
    results = []      # (arr_time, dep_time, flow_type, queuing_delay_s)

    idx = 0
    N = len(events)
    while idx < N or crit_queue or bg_queue:
        next_arrival = events[idx][0] if idx < N else float('inf')
        next_departure = server_free_time if (crit_queue or bg_queue) else float('inf')

        if next_arrival <= next_departure:
            # Process arrival
            t, flow, pid, size = events[idx]
            idx += 1
            pkt = Packet(t, size, flow, pid)
            if flow == 'crit':
                heapq.heappush(crit_queue, (pkt.arr_time, pkt.pid, pkt))
            else:
                heapq.heappush(bg_queue, (pkt.arr_time, pkt.pid, pkt))

            if server_free_time <= t and (crit_queue or bg_queue):
                if crit_queue:
                    _, _, p = heapq.heappop(crit_queue)
                else:
                    _, _, p = heapq.heappop(bg_queue)
                service_time = p.size / LINK_RATE_BPS
                start_time = max(t, server_free_time)
                dep_time = start_time + service_time
                results.append((p.arr_time, dep_time, p.flow_type, dep_time - p.arr_time))
                server_free_time = dep_time
        else:
            # Process departure
            t = next_departure
            if crit_queue or bg_queue:
                if crit_queue:
                    _, _, p = heapq.heappop(crit_queue)
                else:
                    _, _, p = heapq.heappop(bg_queue)
                service_time = p.size / LINK_RATE_BPS
                start_time = max(t, server_free_time)
                dep_time = start_time + service_time
                results.append((p.arr_time, dep_time, p.flow_type, dep_time - p.arr_time))
                server_free_time = dep_time
            else:
                server_free_time = t

    # Add propagation delay
    final = []
    for arr, dep, flow, queuing_delay in results:
        total_delay = queuing_delay + PROP_DELAY_S
        final.append((arr, total_delay, flow))
    return final

# ========== Metrics ==========
def compute_metrics(results):
    crit_delays = [d for (_, d, f) in results if f == 'crit']
    if not crit_delays:
        return None
    crit_delays_ms = [d * 1000 for d in crit_delays]
    max_delay_ms = max(crit_delays_ms)
    min_delay_ms = min(crit_delays_ms)
    jitter_ms = max_delay_ms - min_delay_ms
    return max_delay_ms, min_delay_ms, jitter_ms

def pboo_bound():
    R_min = LINK_RATE_BPS
    sigma = CRIT_BURST_BYTES
    sum_T = PROP_DELAY_S
    bound_s = sigma / R_min + sum_T
    return bound_s * 1000

# ========== Run ==========
if __name__ == "__main__":
    results = run_simulation()
    max_delay, min_delay, jitter = compute_metrics(results)
    bound = pboo_bound()
    print(f"PBOO bound (theoretical): {bound:.2f} ms")
    print(f"Simulated max delay:      {max_delay:.2f} ms")
    print(f"Simulated min delay:      {min_delay:.2f} ms")
    print(f"Jitter (max - min):       {jitter:.2f} ms")
    assert max_delay <= bound + 0.1, "Simulation exceeded bound - check parameters"
