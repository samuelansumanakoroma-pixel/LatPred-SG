import numpy as np
import pandas as pd

# Reproducible seed
np.random.seed(42)

# Simulation parameters
n_packets = 5000
arrival_rate = 100.0   # packets per second (λ)
service_rate = 125.0   # packets per second (μ)

# Generate inter-arrival times (exponential -> Poisson process)
inter_arrival = np.random.exponential(1.0 / arrival_rate, n_packets)
service_times = np.random.exponential(1.0 / service_rate, n_packets)

# Arrival times (cumulative sum)
arrival_times = np.cumsum(inter_arrival)

# M/M/1 queue simulation: departure times and delays
departure_times = np.zeros(n_packets)
queue_delays = np.zeros(n_packets)  # in seconds

for i in range(n_packets):
    if i == 0:
        departure_times[i] = arrival_times[i] + service_times[i]
    else:
        departure_times[i] = max(arrival_times[i], departure_times[i-1]) + service_times[i]
    queue_delays[i] = departure_times[i] - arrival_times[i]   # total delay (queuing + service)

# Convert to milliseconds
latency_ms = queue_delays * 1000.0

# Create feature matrix: arrival rate, queue length, bandwidth (simulated)
# For simplicity, we use lagged latency as features (realistic for time series)
df = pd.DataFrame({
    'packet_id': np.arange(n_packets),
    'arrival_time_s': arrival_times,
    'service_time_s': service_times,
    'true_latency_ms': latency_ms,
    'arrival_rate_estimate': arrival_rate,   # constant in this simulation
    'queue_length': np.cumsum(np.ones(n_packets)) - np.searchsorted(departure_times, arrival_times)  # approximate
})

# Add lag features for better prediction
for lag in [1, 2, 3, 4, 5]:
    df[f'latency_lag_{lag}'] = df['true_latency_ms'].shift(lag)

df = df.dropna().reset_index(drop=True)

# Save dataset
df.to_csv('smart_grid_latency_dataset.csv', index=False)
print(f"Dataset saved: {len(df)} samples")
print(df.head())
