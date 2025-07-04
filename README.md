# Efficient YOLO Models for Litter Detection

This repository implements and evaluates efficient YOLO models for litter detection based on the methodology described in **"Efficient Deep Learning Models for Litter Detection in the Wild"** by Bianco et al. (2024), using the TACO dataset from **TACO: Trash Annotations in Context for Litter Detection** by Pedro F Proença and Pedro Simões (2020)

## 📄 Paper Reference

**Bianco, S., Gaviraghi, E., & Schettini, R. (2024).** *Efficient Deep Learning Models for Litter Detection in the Wild.* 2024 IEEE 8th Forum on Research and Technologies for Society and Industry Innovation (RTSI), 601-606. DOI: 10.1109/RTSI61910.2024.10761805

**Proença, P. F., & Simões, P. (2020).** *TACO: trash annotations in context for litter detection.* CoRR,abs/2003.06975. Retrieved from https://arxiv.org/abs/2003.06975

## 🎯 Project Overview

This project aims to assess the performance of efficient object detection models for litter detection, specifically targeting deployment on resource-constrained devices typically used in citizen science activities (e.g., smartphones with low processing capabilities).

## 📊 Dataset: TACO-1

We use the **TACO (Trash Annotations in Context)** dataset, specifically the TACO-1 variant:

- **Total Images**: 1,500 high-resolution images
- **Annotations**: 4,784 bounding box annotations  
- **Classes**: Single class ("litter") - binary detection task
- **Challenges**: 
  - Very small objects (cigarette butts, bottle caps)
  - Transparent objects (bottles, glass)
  - Realistic outdoor scenarios with uncontrolled conditions
  - High object diversity and complex backgrounds

### Dataset Split
Following Bianco et al. methodology:
- **Training**: 70% (1,050 images)
- **Validation**: 10% (150 images) 
- **Testing**: 20% (300 images)

## 🤖 YOLO Models Evaluated

### Part 1: YOLOv5 & YOLOv8 (Bianco et al. Replication)
- **YOLOv5n** (yolov5nu.pt) - 5.0 MB
- **YOLOv5s** (yolov5su.pt) - 17.7 MB  
- **YOLOv5n6u** (yolov5n6u.pt) - 8.3 MB (1280×1280 input)
- **YOLOv5s6u** (yolov5s6u.pt) - 29.6 MB (1280×1280 input)
- **YOLOv8n** (yolov8n.pt) - 6.0 MB
- **YOLOv8s** (yolov8s.pt) - 21.5 MB

### Part 2: Extended YOLO Evaluation
- **YOLOv9t** - Tiny variant
- **YOLOv9s** - Small variant
- **YOLOv10n** - Nano variant  
- **YOLOv10s** - Small variant
- **YOLOv11n** - Nano variant
- **YOLOv11s** - Small variant
- **YOLOv12n** - Nano variant (if available)
- **YOLOv12s** - Small variant (if available)

## ⚙️ Training Configuration

### Replicating Bianco et al. Setup

**Training Parameters:**
```python
TRAINING_PARAMS = {
    'epochs': 100,
    'imgsz': 640,  # 1280 for 6u variants
    'batch': -1,   # Automatic batch size selection
    'optimizer': 'auto',
    'device': 'cuda' if available else 'cpu',
    'workers': 8,
    'val': True,
    'plots': True
}
```

**Data Augmentation:**
```python
AUGMENTATION_PARAMS = {
    'flipud': 0.5,      # Vertical flip probability
    'degrees': 10.0,    # Rotation range [-10°, +10°] 
    'copy_paste': 0.1   # Copy-paste augmentation
}
```

**Confidence Threshold Optimization:**
- Tested values: [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
- Selected based on highest mAP50 on validation set

## 📈 Evaluation Metrics

1. **mAP50**: Mean Average Precision at IoU threshold 0.50
2. **mAP50-95**: Mean Average Precision averaged over IoU thresholds 0.50-0.95
3. **mAP50 Optimization**: Highest mAP50 after Confidence Threshold Optimization
4. **mAP50-95 Optimization**: Highest mAP50-95 after Confidence Threshold Optimization
5. **Model Size**: In Megabytes (MB)
6. **Inference Speed**: Frames Per Second (FPS)
7. **Training Time**: Hours for convergence

## 🛠️ Repository Structure

```
├── notebooks/
│   ├── part1_yolov5_yolov8.ipynb    # Bianco et al. replication
│   └── part2_extended_yolo.ipynb    # Extended YOLO evaluation
├── src/
│   ├── data_conversion.py           # COCO to YOLO format conversion
│   ├── dataset_splitting.py        # Train/val/test split
│   ├── training_functions.py       # Model training utilities
│   └── evaluation_metrics.py       # Performance evaluation
├── results/
│   ├── part1_results.csv           # Part 1 experimental results
│   ├── part2_results.csv           # Part 2 experimental results
│   └── visualizations/             # Performance plots and charts
├── data/
│   └── taco_yolo/                  # Converted YOLO format dataset
└── README.md
```

## 🚀 Quick Start

### 1. Dataset Preparation
```python
# Convert COCO annotations to YOLO format
python src/data_conversion.py

# Split dataset into train/val/test
python src/dataset_splitting.py
```

### 2. Training
```python
# Train all Part 1 models (YOLOv5 & YOLOv8)
jupyter notebook notebooks/part1_yolov5_yolov8.ipynb

# Train extended YOLO models  
jupyter notebook notebooks/part2_extended_yolo.ipynb
```

### 3. Evaluation
Results are automatically saved to `results/` directory with:
- Performance metrics (CSV format)
- Confidence threshold optimization results
- Inference speed benchmarks
- Model size comparisons

## 📋 Requirements

```
ultralytics>=8.2.7
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
scikit-learn>=1.0.0
```

## 💡 Key Innovations

1. **Confidence Threshold Optimization**: Systematic tuning improves mAP50
2. **Efficient Model Focus**: Prioritizing nano and small variants for edge deployment
3. **Comprehensive Evaluation**: Including inference speed and model size trade-offs
4. **Citizen Science Applications**: Targeting smartphone-deployable models

## 🔬 Research Impact

This work demonstrates that efficient YOLO models can achieve **state-of-the-art performance** on litter detection while being suitable for deployment on resource-constrained devices, enabling:

- Large-scale citizen science initiatives
- Real-time litter monitoring systems  
- Mobile applications for environmental conservation
- Edge computing deployments

## 📞 Contact

For questions about this implementation or the research methodology, please open an issue in this repository.

## 📚 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{bianco2024efficient,
  title={Efficient Deep Learning Models for Litter Detection in the Wild},
  author={Bianco, Simone and Gaviraghi, Elia and Schettini, Raimondo},
  booktitle={2024 IEEE 8th Forum on Research and Technologies for Society and Industry Innovation (RTSI)},
  pages={601--606},
  year={2024},
  organization={IEEE},
  doi={10.1109/RTSI61910.2024.10761805}
}
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
'''

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This repository supports the advancement of automated litter detection for environmental conservation through efficient deep learning models.*
