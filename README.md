# Hardware-Aware DNN Optimization for Edge Deployment

## 📌 Overview
This project focuses on optimizing deep neural networks (CNNs and Vision Transformers) for resource-constrained edge devices using hardware-aware techniques.

## 🚀 Key Features
- Platform-aware model compression (Quantization, Pruning, Width/Depth Shrinking)
- Latency and memory optimization for FPGA/SoC and GPU
- Deployment on Xilinx ZCU102 and NVIDIA Jetson

## 🏗️ Approach
1. Baseline model training (ResNet / MobileNet / ViT)
2. Apply hardware-aware pruning (width shrinking)
3. Depth reduction via layer merging
4. Quantization-aware training (QAT)
5. Deployment using TensorRT / FPGA pipeline

## ⚙️ Tech Stack
- PyTorch, CUDA, TensorRT
- Xilinx FPGA (ZCU102)
- Python, NumPy

## 📊 Results
| Metric        | Before | After |
|--------------|--------|-------|
| Latency      | 120 ms | 45 ms |
| Memory       | 200 MB | 80 MB |
| Accuracy     | 92%    | 90%   |

## 📁 Structure
