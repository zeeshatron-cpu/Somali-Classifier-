# Somali Name Classifier

A comprehensive AI system built with LLaMA to distinguish between Somali and non-Somali names. This system includes multiple approaches for training and inference, optimized for your existing dataset and infrastructure.

## Features

- **Multiple Training Approaches**: Support for both classification and LoRA fine-tuning
- **Flexible Inference**: Works with existing trained models and new training
- **Batch Processing**: Efficient batch prediction for large datasets
- **Comprehensive Evaluation**: Detailed metrics and analysis
- **Easy Integration**: Simple command-line interface

## Files Overview

### Core Scripts

1. **`somali_name_classifier.py`** - Complete training and inference system
2. **`advanced_somali_classifier.py`** - Advanced system with LoRA support
3. **`train_somali_classifier.py`** - Simple training script
4. **`inference_somali_classifier.py`** - Simple inference script

### Your Existing Files

- **`synthetic_dataset.csv`** - Your training dataset (9,183 samples)
- **`somali_classifier_full_dataset.csv`** - Full dataset (11,970 samples)
- **`somali-lora/`** - Your existing LoRA model
- **`tinyllama_lora_model/`** - Another trained model

## Quick Start

### 1. Test with Your Existing Model

```bash
# Test your existing LoRA model
python advanced_somali_classifier.py --model_path ./somali-lora/checkpoint-250 --mode predict --names "AHMED MOHAMED" "JOHN SMITH" "FATIMA HUSSEIN"

# Or use the simple inference script
python inference_somali_classifier.py --model_path ./somali-lora/checkpoint-250 --mode single --name "AHMED MOHAMED"
```

### 2. Train a New Model

```bash
# Train a new classification model
python train_somali_classifier.py --data synthetic_dataset.csv --output_dir ./new_somali_model --epochs 5

# Or use the comprehensive training script
python somali_name_classifier.py --mode train --data synthetic_dataset.csv --model_path ./new_somali_model
```

### 3. Evaluate on Your Dataset

```bash
# Evaluate your existing model
python advanced_somali_classifier.py --model_path ./somali-lora/checkpoint-250 --mode evaluate --dataset synthetic_dataset.csv

# Or evaluate a new model
python inference_somali_classifier.py --model_path ./new_somali_model --mode csv --csv_path synthetic_dataset.csv
```

## Detailed Usage

### Training a New Model

#### Option 1: Simple Training Script
```bash
python train_somali_classifier.py \
    --data synthetic_dataset.csv \
    --model_name microsoft/DialoGPT-medium \
    --output_dir ./somali_classifier_model \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

#### Option 2: Comprehensive Training Script
```bash
python somali_name_classifier.py \
    --mode train \
    --data synthetic_dataset.csv \
    --model_path ./somali_classifier_model \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

### Making Predictions

#### Single Name Prediction
```bash
python inference_somali_classifier.py \
    --model_path ./somali_classifier_model \
    --mode single \
    --name "AHMED MOHAMED" \
    --show_probs
```

#### Batch Prediction
```bash
python inference_somali_classifier.py \
    --model_path ./somali_classifier_model \
    --mode batch \
    --names "AHMED MOHAMED" "JOHN SMITH" "FATIMA HUSSEIN" "EMILY JOHNSON" \
    --show_probs
```

#### CSV File Prediction
```bash
python inference_somali_classifier.py \
    --model_path ./somali_classifier_model \
    --mode csv \
    --csv_path synthetic_dataset.csv \
    --name_column "Full Name" \
    --output results.csv
```

### Advanced Features

#### Using LoRA Models
```bash
python advanced_somali_classifier.py \
    --model_path ./somali-lora/checkpoint-250 \
    --mode predict \
    --names "AHMED MOHAMED" "JOHN SMITH" \
    --quantization  # Use 4-bit quantization for memory efficiency
```

#### Comprehensive Evaluation
```bash
python advanced_somali_classifier.py \
    --model_path ./somali_classifier_model \
    --mode evaluate \
    --dataset synthetic_dataset.csv \
    --output detailed_results.csv \
    --batch_size 16
```

## Dataset Format

Your datasets should have the following format:

```csv
Full Name,label
AHMED MOHAMED,1
JOHN SMITH,0
FATIMA HUSSEIN,1
EMILY JOHNSON,0
```

Where:
- `Full Name` (or `name`): The name to classify
- `label`: 1 for Somali names, 0 for non-Somali names

## Model Performance

Based on your existing dataset:
- **Total samples**: 9,183 (synthetic) / 11,970 (full)
- **Class distribution**: Mixed Somali and non-Somali names
- **Expected accuracy**: 85-95% with proper training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use smaller batch size
   python train_somali_classifier.py --batch_size 8
   
   # Or use quantization
   python advanced_somali_classifier.py --quantization
   ```

2. **Model Loading Errors**
   ```bash
   # Check if model path exists
   ls -la ./somali-lora/checkpoint-250/
   
   # Try different model paths
   python advanced_somali_classifier.py --model_path ./tinyllama_lora_model
   ```

3. **Dataset Format Issues**
   ```bash
   # Check your CSV format
   head -5 synthetic_dataset.csv
   
   # Use correct column name
   python inference_somali_classifier.py --name_column "Full Name"
   ```

### Performance Optimization

1. **For Training**:
   - Use smaller batch sizes if you have limited GPU memory
   - Reduce learning rate for more stable training
   - Increase epochs for better convergence

2. **For Inference**:
   - Use batch processing for multiple names
   - Enable quantization for memory efficiency
   - Use GPU if available

## Example Results

```
SOMALI NAME CLASSIFICATION RESULTS
============================================================
AHMED MOHAMED           --> Somali       (confidence: 0.923)
JOHN SMITH              --> Not Somali   (confidence: 0.891)
FATIMA HUSSEIN          --> Somali       (confidence: 0.945)
EMILY JOHNSON           --> Not Somali   (confidence: 0.876)
```

## Next Steps

1. **Train a new model** with your full dataset
2. **Evaluate performance** on different name types
3. **Fine-tune hyperparameters** for better accuracy
4. **Deploy the model** for production use

## Support

If you encounter any issues:
1. Check the error messages carefully
2. Verify your dataset format
3. Ensure all dependencies are installed
4. Try different model paths or parameters


