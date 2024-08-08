# voice matching

### Overview

This voice matching system is designed to compare a test audio file against a set of predefined reference audio files. It aims to identify the closest matching voice by analyzing various audio features and calculating the similarities between them.

### Features

1. **Audio Upload:**
   - Users can upload a test voice file for comparison. This file serves as the input for the voice matching process.

2. **Audio Feature Extraction:**
   - The system extracts key audio features from both the test file and the reference files. These features include:
     - **MFCC (Mel-frequency cepstral coefficients):** Captures the power spectrum of the audio.
     - **Chroma:** Represents the twelve different pitch classes in music.
     - **Zero-Crossing Rate (ZCR):** Measures the rate at which the signal changes sign.
     - **Energy Envelope:** Represents the energy level of the audio signal over time.

3. **Similarity Metrics:**
   - The system uses several metrics to evaluate the similarity between the test voice and the reference voices:
     - **Euclidean Distance:** Measures the straight-line distance between feature vectors.
     - **Manhattan Distance:** Calculates the sum of absolute differences between feature vectors.
     - **Cosine Similarity:** Measures the cosine of the angle between feature vectors, indicating their orientation.
     - **Kendall’s Tau:** Evaluates the correlation between the feature vectors.

4. **Best Match Identification:**
   - Based on the calculated similarities, the system identifies the reference voice that most closely matches the test voice. It uses cosine similarity as a primary criterion for determining the best match.

5. **Visual Comparison:**
   - The system generates plots that compare the features of the test voice with those of the best-matched reference voice. These plots include:
     - **MFCC Comparison**
     - **Chroma Comparison**
     - **ZCR Comparison**
     - **Energy Envelope Comparison**
   - The visualizations help users visually assess the similarity between the voices.

6. **Results Presentation:**
   - Results are displayed in an HTML table, showing the calculated distances and similarities for each feature type. 
   - The system highlights the best match along with its similarity score.

### How It Works

1. **Upload Test Audio:**
   - Users upload their test voice file through the Gradio interface.

2. **Feature Extraction:**
   - The system processes the test audio file and reference files to extract the relevant audio features.

3. **Distance Calculation:**
   - It calculates various distances and similarities between the features of the test and reference voices.

4. **Best Match Determination:**
   - The system identifies the reference voice that best matches the test voice based on the calculated similarities.

5. **Visualization and Results:**
   - Visual plots comparing the features are generated.
   - A summary of the comparison results is presented, highlighting the closest match and similarity metrics.

This voice matching system is useful for applications such as voice identification, forensic analysis, and any scenario where comparing audio recordings is necessary. It leverages advanced audio processing techniques to provide accurate and meaningful comparisons.