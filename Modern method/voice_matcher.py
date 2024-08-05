import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import librosa
import os

def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    audio, sr = librosa.load(file_path, sr=None)
    print(f'Loaded audio from {file_path}')
    return audio, sr

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
    print('Extracted features:')
    for key, value in features.items():
        print(f'{key} shape: {value.shape}')
    return features

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
        print(f'Normalized {key}: mean={mean.mean():.2f}, std={std.mean():.2f}')
    return normalized_features

def train_gmm(features, n_components=8, reg_covar=1e-4):
    models = {}
    for key, feature in features.items():
        if feature.size == 0:
            raise ValueError(f'Feature {key} is empty.')
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', reg_covar=reg_covar, init_params='kmeans', random_state=42)
        gmm.fit(feature)
        models[key] = gmm
        print(f'Trained GMM for {key}: Converged={gmm.converged_}, Means={gmm.means_}')
    return models

def score_gmm(model, feature):
    if feature.size == 0:
        raise ValueError('Feature is empty.')
    log_likelihood = model.score_samples(feature)
    mean_log_likelihood = np.mean(log_likelihood)
    return mean_log_likelihood

def prepare_training_data(train_files):
    X = []
    y = []
    for i, file in enumerate(train_files):
        print(f'Processing training file {file}')
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

def determine_threshold(train_scores, percentile=45):
    threshold = np.percentile(train_scores, 100 - percentile)
    print(f'Determined threshold for low similarity: {threshold:.2f}')
    return threshold

def match_test_to_train(test_file, train_files):
    if len(train_files) == 0:
        raise ValueError('No training files provided.')

    train_models = {}
    train_scores = []

    for file in train_files:
        print(f'Loading training file {file}')
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
    print(f'Scaled training features:\n{X_train_scaled}')

    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_scaled, np.arange(len(train_files)))
    print('Trained logistic regression model')

    print(f'\nProcessing test file {test_file}')
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
    print(f'Predicted scores (flattened): {predicted_scores}')

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
        else:
            print(f'{feature} scores not available.')

    print(f'\nFeature Weights:')
    feature_names = ['mfcc', 'imfcc', 'lfcc', 'pncc']
    weights = logistic_model.coef_[0]
    for feature, weight in zip(feature_names, weights):
        print(f'{feature}: {weight:.4f}')

# Define the list of training files
train_files = [
    '/content/1eDs54829Hec_fungua.wav',
    '/content/1eDs54829Hec_good_morning.wav',
    '/content/1eDs54829Hec_good_afternoon.wav',
    '/content/1eDs54829Hec_ni_mimi.wav',
    '/content/1eDs54829Hec_hello.wav'
]

# Define the test file
test_file = '/content/1e3391d5.wav'

# Call the function with predefined train files and test file
match_test_to_train(test_file, train_files)
