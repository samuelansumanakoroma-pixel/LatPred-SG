import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
n_packets = 10000
arrival_rate_normal = 100   # packets/sec
arrival_rate_fault = 800    # packets/sec during fault
service_rate = 125          # packets/sec

# Time array (ms)
time_ms = np.arange(n_packets) * 10  # each packet every 10ms on average

# Fault events (indices)
fault_intervals = [(2000, 2100), (5000, 5150), (8000, 8200)]
fault_active = np.zeros(n_packets, dtype=bool)
for start, end in fault_intervals:
    fault_active[start:end] = True

queue_delay_ms = np.zeros(n_packets)
packet_loss = np.zeros(n_packets)

for i in range(n_packets):
    lam = arrival_rate_fault if fault_active[i] else arrival_rate_normal
    if lam < service_rate:
        delay_sec = 1.0 / (service_rate - lam)
    else:
        # Overload: delay increases and loss occurs
        delay_sec = 1.0 / max(0.001, service_rate - lam + 0.001)
        packet_loss[i] = min(0.5, (lam - service_rate) / service_rate * 0.5)
    queue_delay_ms[i] = delay_sec * 1000 + np.random.normal(0, 0.5)
    queue_delay_ms[i] = max(0.5, queue_delay_ms[i])

propagation_ms = 1.5
processing_ms = 0.8
total_latency_ms = queue_delay_ms + propagation_ms + processing_ms
traffic_load = np.where(fault_active, arrival_rate_fault/service_rate, arrival_rate_normal/service_rate)

df = pd.DataFrame({
    'time_ms': time_ms,
    'traffic_load': traffic_load,
    'queue_delay_ms': queue_delay_ms,
    'packet_loss': packet_loss,
    'total_latency_ms': total_latency_ms,
    'fault_event': fault_active.astype(int)
})

df.to_csv('smartgrid_dataset.csv', index=False)
print(f"Saved {len(df)} rows to smartgrid_dataset.csv")
