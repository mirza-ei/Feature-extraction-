# Speaker Verification System

## Overview

This project is a Speaker Verification System that uses a deep learning model to identify the most similar audio file from a set of pre-defined training files. It leverages audio processing techniques and a VGG16-based feature extractor to compare audio files.

## Components

1. **Audio Processing**:
   - Converts audio files into mel spectrograms using `librosa`.
   - Resizes the spectrogram to fit the input shape required by the VGG16 model.

2. **Feature Extraction**:
   - Utilizes a VGG16-based model to extract features from the mel spectrograms.
   - The VGG16 model is fine-tuned to output a feature vector suitable for similarity comparison.

3. **Feature Normalization**:
   - Normalizes the extracted features using `StandardScaler` to ensure consistency and comparability.

4. **Similarity Measurement**:
   - Computes cosine distances between the feature vectors of the uploaded audio and the training files to find the most similar file.

5. **User Interface**:
   - Employs `gradio` to create a simple web interface where users can upload an audio file and receive the most similar file from the training set along with a similarity score.

## Usage

1. **Model Setup**:
   - The VGG16 model is loaded with weights from ImageNet and adapted for feature extraction.

2. **Training Files**:
   - Specify the paths to the training audio files. These files are used to build the reference feature set.

3. **Feature Extraction**:
   - Extract and normalize features from the training files.

4. **Verification**:
   - Upload an audio file through the Gradio interface.
   - The system processes the audio, extracts features, and compares them to the training set.
   - Displays the path of the most similar training file and the similarity score.

## Example Workflow

1. **Preprocess**:
   - Convert audio to mel spectrogram.
   - Resize and adjust the spectrogram for the model.

2. **Feature Extraction**:
   - Use VGG16-based model to get feature vectors.

3. **Comparison**:
   - Normalize features and compute cosine distances.

4. **User Interface**:
   - Gradio interface allows users to upload an audio file and view results.

## Launching the Interface

To start the web interface:

```python
iface.launch()
```

This command launches the Gradio app, enabling users to interact with the speaker verification system via their web browser.

Feel free to modify the paths to the training files and adjust other parameters as needed to fit your specific use case.