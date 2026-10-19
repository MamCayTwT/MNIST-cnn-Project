# MNIST Digit Classification with Deep CNN

A PyTorch implementation of a deep CNN for MNIST digit classification.

## Project Structure

- `Working with MINIST.ipynb` — End-to-end training & testing pipeline
- `Inference/` — Run predictions on new data without retraining
- `Data/` — Custom MNIST dataset loader
- `Dataset/` — CSV files (train.csv, test.csv)
- `outputs/` — Saved model & submission file
- `requirements.txt` — Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Training

Open `Working with MINIST.ipynb` and run all cells to train the model.

## Inference (predict on new data)

Open `Inference/` notebook and run all cells. Make sure `outputs/best_model.pth` exists.

## Model Architecture

- 3 convolutional layers (32 → 64 → 128 channels) with ReLU and MaxPooling
- Fully connected layers with Dropout (0.25)
- Total parameters: ~1.2M

## Results

| Metric | Value |
|--------|-------|
| Train Accuracy | ~99% |
| Validation Accuracy | ~98% |

## License

MIT License