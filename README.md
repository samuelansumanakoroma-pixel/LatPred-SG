# LatPred-SG: NS-3 Simulation Framework for Smart Grid Latency Prediction

This repository contains the NS-3 simulation code and transformer-based prediction model used in the thesis:

**"Transformer-Based Method for Reducing Latency in Smart Grid Communication"**
Author: Koroma Samuel Ansumana
Supervisor: Professor Wang Hao
Dalian University of Technology

## Overview

This project implements a hierarchical HAN-NAN-WAN smart grid communication network in NS-3 and trains a Transformer model (LatPred-SG) to predict deterministic delay bounds, tail latency (99th percentile), and worst-case delays.

### Key Features

- **NS-3 Simulation**: 50 smart meters, 5 aggregation gateways, 2 substations, 1 control center.
- **Traffic Models**: Periodic monitoring (CBR), fault-induced bursts (ON-OFF), background cross-traffic.
- **Queuing**: Strict priority queuing (IEEE 802.1Q) for fault messages.
- **Transformer Model**: Hybrid analytical (G/D/1) + residual learning for latency prediction.
- **Metrics**: MAE, RMSE, 99th percentile tail latency, worst-case delay, violation probability.

## Requirements

### NS-3
- NS-3 version 3.38 or later
- Ubuntu 20.04+ (or WSL2 on Windows)

### Python
- Python 3.8+
- PyTorch 1.12+
- NumPy, Pandas, Matplotlib

## Installation

### 1. Install NS-3

```bash
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev
./ns3 configure --build-profile=release --enable-examples --enable-tests
./ns3 build
