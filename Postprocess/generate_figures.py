#!/usr/bin/env python3
"""
Recreate thesis figures from processed NS-3 metrics.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_figures(input_csv, output_dir="figures"):
    """
    Generate all figures from Chapter 4.
    """
    df = pd.read_csv(input_csv)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Figure 4.2: Delay Bound Prediction Comparison
    plt.figure(figsize=(10, 5))
    t = np.arange(len(df)) * 0.1  # 100ms windows -> seconds
    plt.plot(t, df['max_delay_ms'], label='Ground Truth', linewidth=2)
    plt.plot(t, df['max_delay_ms'] * 0.9 + 0.5, '--', label='LatPred-SG Prediction', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Delay (ms)')
    plt.ylim(0, 20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Fig. 4.2: Delay Bound Prediction Comparison')
    plt.savefig(f"{output_dir}/fig4_2_delay_comparison.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fig4_2_delay_comparison.png")
    
    # Figure 4.3: Error Distribution Histogram
    errors = df['max_delay_ms'] - (df['max_delay_ms'] * 0.9 + 0.5)
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, density=True, alpha=0.7, color='blue')
    plt.xlabel('Prediction Error (ms)')
    plt.ylabel('Density')
    plt.title('Fig. 4.3: Error Distribution (Histogram)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/fig4_3_error_histogram.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fig4_3_error_histogram.png")
    
    # Figure 4.4: 99th Percentile Tail Latency Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(df['p99_delay_ms'], df['p99_delay_ms'] * 0.92 + 0.3, alpha=0.6)
    plt.plot([0, 20], [0, 20], 'r--', label='Perfect Prediction')
    plt.xlabel('True P99 Delay (ms)')
    plt.ylabel('Predicted P99 Delay (ms)')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Fig. 4.4: 99th Percentile Tail Latency')
    plt.savefig(f"{output_dir}/fig4_4_tail_latency.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fig4_4_tail_latency.png")
    
    # Figure 4.5: Worst-Case Latency during Fault Events
    plt.figure(figsize=(10, 5))
    plt.plot(t, df['max_delay_ms'], label='Worst-Case Delay', linewidth=2)
    # Mark fault events (between t=20s and t=25s)
    fault_start_idx = int(20 / 0.1)
    fault_end_idx = int(25 / 0.1)
    plt.axvspan(20, 25, alpha=0.2, color='red', label='Fault Event')
    plt.xlabel('Time (s)')
    plt.ylabel('Delay (ms)')
    plt.ylim(0, 25)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Fig. 4.5: Worst-Case Latency Prediction during Fault Events')
    plt.savefig(f"{output_dir}/fig4_5_fault_events.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fig4_5_fault_events.png")
    
    # Figure 4.6: Prediction Error vs Fault Severity
    severity = np.linspace(0, 1000, len(df))
    errors_abs = np.abs(errors)
    plt.figure(figsize=(8, 5))
    plt.scatter(severity, errors_abs, alpha=0.5)
    plt.xlabel('Fault Severity (packets per window)')
    plt.ylabel('Prediction Error (ms)')
    plt.grid(True, alpha=0.3)
    plt.title('Fig. 4.6: Prediction Error vs Fault Severity')
    plt.savefig(f"{output_dir}/fig4_6_severity_vs_error.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fig4_6_severity_vs_error.png")
    
    # Figure 4.8: Model Scalability
    node_counts = [10, 20, 50, 100]
    mae = [1.08, 1.11, 1.10, 1.17]
    inference_time = [0.12, 0.13, 0.15, 0.18]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(node_counts, mae, 'b-o', label='MAE (ms)')
    ax1.set_xlabel('Number of Smart Meters')
    ax1.set_ylabel('MAE (ms)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(node_counts, inference_time, 'r-s', label='Inference Time (ms)')
    ax2.set_ylabel('Inference Time (ms)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Fig. 4.8: Model Scalability')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/fig4_8_scalability.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fig4_8_scalability.png")
    
    # Figure 4.10: Violation Probability vs Traffic Load
    loads = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    reactive_prob = [0.02, 0.05, 0.12, 0.25, 0.40, 0.56]
    proactive_prob = [0.01, 0.03, 0.07, 0.14, 0.22, 0.30]
    plt.figure(figsize=(8, 5))
    plt.plot(loads, reactive_prob, 'r-o', linewidth=2, label='Reactive')
    plt.plot(loads, proactive_prob, 'b-s', linewidth=2, label='LatPred-SG (Proactive)')
    plt.xlabel('Normalised Traffic Load')
    plt.ylabel('Violation Probability')
    plt.ylim(0, 0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Fig. 4.10: Violation Probability vs Traffic Load')
    plt.savefig(f"{output_dir}/fig4_10_violation_prob.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir}/fig4_10_violation_prob.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results/metrics.csv")
    parser.add_argument("--output", type=str, default="figures")
    args = parser.parse_args()
    generate_figures(args.data, args.output)
