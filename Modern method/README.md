


# Audio Matching System

---



## Overview

This project implements an audio matching system using Gaussian Mixture Models (GMMs) and Logistic Regression. The system extracts audio features, trains GMMs to model these features, and uses logistic regression to classify or rank the similarity between test and training audio files.

## Requirements

- Python 3.x
- `numpy`
- `scikit-learn`
- `librosa`
- `os`

## Installation

Install the required packages using pip:

```bash
pip install numpy scikit-learn librosa
```

## Code Description

### 1. **Loading Audio**

**Function**: `load_audio(file_path)`

Loads an audio file and returns the audio data and sample rate.

```python
def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    audio, sr = librosa.load(file_path, sr=None)
    print(f'Loaded audio from {file_path}')
    return audio, sr
```

### 2. **Feature Extraction**

**Function**: `extract_features(audio, sr)`

Extracts audio features using MFCC, IMFCC, LFCC, and PNCC.

```python
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    imfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, dct_type=3)
    lfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, htk=True)
    pncc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, lifter=10)
    features = {
        'mfcc': mfcc.T,
        'imfcc': imfcc.T,
        'lfcc': lfcc.T,
        'pncc': pncc.T
    }
    return features
```

### 3. **Normalization**

**Function**: `normalize_features(features)`

Normalizes extracted features by removing the mean and scaling to unit variance.

```python
def normalize_features(features):
    normalized_features = {}
    for key, feature in features.items():
        if feature.size == 0:
            raise ValueError(f'Feature {key} is empty.')
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        if np.any(std == 0):
            raise ValueError(f'Standard deviation for feature {key} is zero.')
        normalized_features[key] = (feature - mean) / std
    return normalized_features
```

### 4. **GMM Training**

**Function**: `train_gmm(features, n_components=8, reg_covar=1e-4)`

Trains a Gaussian Mixture Model for each feature type.

```python
def train_gmm(features, n_components=8, reg_covar=1e-4):
    models = {}
    for key, feature in features.items():
        if feature.size == 0:
            raise ValueError(f'Feature {key} is empty.')
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', reg_covar=reg_covar, init_params='kmeans', random_state=42)
        gmm.fit(feature)
        models[key] = gmm
    return models
```

### 5. **Scoring**

**Function**: `score_gmm(model, feature)`

Calculates the log-likelihood of features given the GMM.

```python
def score_gmm(model, feature):
    if feature.size == 0:
        raise ValueError('Feature is empty.')
    log_likelihood = model.score_samples(feature)
    mean_log_likelihood = np.mean(log_likelihood)
    return mean_log_likelihood
```

**Log-Likelihood Explanation**:
- **Log-Likelihood** is a measure of how well the GMM explains the observed data. For each data point (feature vector), it calculates the logarithm of the probability density function of the GMM.
- **Purpose**: It helps determine how likely the observed features are under the trained GMM. Higher values indicate that the feature vector is well-explained by the model.

### 6. **Preparing Training Data**

**Function**: `prepare_training_data(train_files)`

Prepares training data by extracting features, normalizing them, training GMMs, and computing scores.

```python
def prepare_training_data(train_files):
    X = []
    y = []
    for i, file in enumerate(train_files):
        audio, sr = load_audio(file)
        features = extract_features(audio, sr)
        normalized_features = normalize_features(features)
        models = train_gmm(normalized_features)
        scores = [score_gmm(models[key], normalized_features[key]) for key in ['mfcc', 'imfcc', 'lfcc', 'pncc']]
        X.append(scores)
        y.append(i)
    X = np.array(X)
    y = np.array(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain at least 2 classes. Only one class found.")
    return X, y
```

### 7. **Threshold Determination**

**Function**: `determine_threshold(train_scores, percentile=45)`

Determines a threshold for similarity based on percentile of training scores.

```python
def determine_threshold(train_scores, percentile=45):
    threshold = np.percentile(train_scores, 100 - percentile)
    return threshold
```

### 8. **Matching Test to Train**

**Function**: `match_test_to_train(test_file, train_files)`

Matches a test file to the most similar training file using the trained GMMs and logistic regression.

```python
def match_test_to_train(test_file, train_files):
    if len(train_files) == 0:
        raise ValueError('No training files provided.')
    train_models = {}
    train_scores = []
    for file in train_files:
        train_audio, train_sr = load_audio(file)
        train_features = extract_features(train_audio, train_sr)
        normalized_train_features = normalize_features(train_features)
        models = train_gmm(normalized_train_features)
        scores = [score_gmm(models[key], normalized_train_features[key]) for key in ['mfcc', 'imfcc', 'lfcc', 'pncc']]
        train_scores.append(scores)
        train_models[file] = models
    train_scores = np.array(train_scores)
    threshold = determine_threshold(train_scores.flatten(), percentile=54)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_scores)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_scaled, np.arange(len(train_files)))
    test_audio, test_sr = load_audio(test_file)
    test_features = extract_features(test_audio, test_sr)
    normalized_test_features = normalize_features(test_features)
    test_scores = []
    for file in train_files:
        models = train_models[file]
        scores = [score_gmm(models[key], normalized_test_features[key]) for key in ['mfcc', 'imfcc', 'lfcc', 'pncc']]
        test_scores.append(scores)
    test_scores = np.array(test_scores)
    test_scores_scaled = scaler.transform(test_scores)
    predicted_scores = logistic_model.decision_function(test_scores_scaled)
    predicted_scores = predicted_scores.flatten()
    if len(predicted_scores) != len(train_files):
        print(f'Error: Mismatch in number of predicted scores ({len(predicted_scores)}) and training files ({len(train_files)}).')
        return
    best_match_index = np.argmax(predicted_scores)
    if best_match_index >= len(train_files):
        print(f'Error: Best match index {best_match_index} is out of bounds.')
        return
    best_match_score = predicted_scores[best_match_index]
    if best_match_score < threshold:
        print(f'For {test_file}, the best match {train_files[best_match_index]} has insufficient similarity (score: {best_match_score:.2f}).')
        print('No sufficient match found.')
    else:
        best_match = train_files[best_match_index]
        print(f'For {test_file}, the best match {best_match} has sufficient similarity (score: {best_match_score:.2f}).')
    print(f'\nTest File Scores:')
    for feature in ['mfcc', 'imfcc', 'lfcc', 'pncc']:
        index = ['mfcc', 'imfcc', 'lfcc', 'pncc'].index(feature)
        if len(test_scores) > 0:
            print(f'{feature} for test file {test_file}: {test_scores[0][index]:.2f}')
    print(f'\nFeature Weights:')
    feature_names = ['mfcc', 'imfcc', 'lfcc', 'pncc']
    weights = logistic_model.coef_[0]
    for feature, weight in zip(feature_names, weights):
        print(f'{feature}: {weight:.4f}')
```

**Logistic Regression Explanation**:

- **Logistic Regression** is used to classify the test file by comparing it against the training files. It finds a linear decision boundary in the feature space.


- **Decision Function**: `decision_function` provides the distance of the samples from the decision boundary. Higher values indicate a stronger match.

## How It Works

1. **Load Audio**: Audio files are loaded and their features are extracted.

2. **Extract Features**: Audio features are extracted using MFCC, IMFCC, LFCC, and PNCC.

3. **Normalize Features**: Features are normalized for consistency.

4. **Train GMM**: Gaussian Mixture Models are trained for each feature type.

5. **Compute Scores**: Scores for each feature are computed using the GMM.

6. **Train Logistic Regression**: Logistic Regression is trained using the scores.

7. **Match Test to Train**: The test file is compared to the training files using the trained model and scores.




