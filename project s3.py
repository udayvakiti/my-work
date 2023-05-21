# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:09:33 2023

@author: udaykiranreddyvakiti
"""
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
import os

# Define a function for filtering and smoothing the data
def preprocess_data(raw_data):
    # Apply a bandpass filter to remove any high-frequency noise
    b, a = signal.butter(3, [10, 100], btype='bandpass', fs=1000)
    filtered_data = signal.filtfilt(b, a, raw_data, axis=0)
    
    # Apply a moving average filter to smooth the data
    smoothed_data = np.convolve(filtered_data, np.ones((10,))/10, mode='valid')
    
    # Normalize the data to a standard scale
    normalized_data = (smoothed_data - np.mean(smoothed_data)) / np.std(smoothed_data)
    
    return normalized_data

# Define a function for segmenting the data into smaller chunks
def segment_data(data, window_size):
    num_windows = int(len(data) / window_size)
    segments = np.zeros((num_windows, window_size))
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        segments[i,:] = data[start:end]
    
    return segments

# Define a function for extracting features from data segments
def extract_features(data):
    # Compute the root mean square (RMS) current
    rms_current = np.sqrt(np.mean(data**2, axis=1))
    
    # Compute the peak current
    peak_current = np.max(data, axis=1)
    
    # Compute the total harmonic distortion (THD)
    fft_data = np.fft.fft(data)
    harmonic_indices = np.arange(1, 10)
    harmonic_amps = np.abs(fft_data[:, harmonic_indices])
    total_amp = np.sum(np.abs(fft_data), axis=1)
    thd = np.sqrt(np.sum(harmonic_amps**2, axis=1)) / total_amp
    
    return np.stack((rms_current, peak_current, thd), axis=1)

# Define a function for detecting anomalies using K-means clustering
def detect_anomalies(data, n_clusters=0):
    # Fit a K-means model to the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    
    # Compute the distance of each data point from its nearest cluster center
    distances = kmeans.transform(data)
    min_distances = np.min(distances, axis=1)
    
    # Set the anomaly threshold to the 95th percentile of the minimum distances
    threshold = np.percentile(min_distances, 95)
    
    # Identify any data points that are above the threshold
    anomalies = np.where(min_distances > threshold)[0]
    
    return anomalies

# Define a function for classifying defects using One-Class SVM
def classify_defects(data, anomaly_indices):
    # Train a One-Class SVM model on the non-anomalous data
    non_anomalous_data = np.delete(data, anomaly_indices, axis=0)
    svm = OneClassSVM(kernel='rbf').fit(non_anomalous_data)
    
    # Classify each segment as anomalous or non-anomalous
    classifications = svm.predict(data)
    
    # Convert the classifications to an array of boolean values
    is_defective = classifications == -1
    
    return is_defective

# Set the path to the directory containing the text files
path = "C:\\Users\\udaykiranreddyvakiti\\s3Files"

# Loop through all files in the directory
for file_name in os.listdir(path):
    if file_name.endswith(".txt"):
        # Read the file into a 1D array
      data = np.loadtxt(file_name, delimiter=',')

# find rows with empty strings
      empty_rows = np.where(data == '')[0]

# remove rows with empty strings
      data = np.delete(data, empty_rows, axis=0)

# reshape data
      data_reshaped = data
        
        # Save the reshaped data as a new file
      new_file_name = file_name.replace(".txt", "_reshaped.txt")
      np.savetxt(os.path.join(path, new_file_name), data_reshaped, delimiter=",")


        # Preprocess the data by splitting the lines and converting to floats
data = []
for line in data:
    values = line.strip().split(',')
    # Skip over any empty values
    values = [float(v) if v != '' else None for v in values]
    if None in values:
        continue # Skip over any lines with empty values
    data.append(values)
   

# Segment the data into windows
window_size = 100 # set the window size
segments = []
for i in range(0, len(data), window_size):
    window = data[i:i+window_size]
    if len(window) == window_size:
        segments.append(window)
    num_rows = 10
    num_cols = int(len(data) / num_rows)

# Reshape your data into a 2D array
    segments = np.reshape(segments, (num_rows, num_cols))
     
# Extract features from the segments
features = []
for segment in segments:
    feature = extract_features(segment)
    features.append(feature)
    num_rows = 10
    num_cols = int(len(data) / num_rows)

# Reshape your data into a 2D array
    features = np.reshape(features, (num_rows, num_cols))
# Detect anomalies in the features
anomalies = detect_anomalies(features)

# Classify defects in the segments
defects = classify_defects(segments, anomalies)



from scipy import signal

# Load the raw current data into a pandas DataFrame
raw_data = pd.read_csv('motor_current_data.csv')

# Apply a bandpass filter to remove any high-frequency noise
b, a = signal.butter(3, [10, 100], btype='bandpass', fs=1000)
filtered_data = signal.filtfilt(b, a, raw_data, axis=0)

# Apply a moving average filter to smooth the data
smoothed_data = np.convolve(filtered_data, np.ones((10,))/10, mode='valid')

# Normalize the data to a standard scale
normalized_data = (smoothed_data - np.mean(smoothed_data)) / np.std(smoothed_data)
from scipy.signal import welch

# Define a function for extracting features from a segment of current data
def extract_features(data_segment):
    # Compute the mean and standard deviation of the current readings
    mean_current = np.mean(data_segment)
    std_current = np.std(data_segment)
    
    # Compute the power spectral density of the current readings
    f, psd = welch(data_segment, fs=1000)
    psd_mean = np.mean(psd)
    psd_std = np.std(psd)
    
    return [mean_current, std_current, psd_mean, psd_std]

# Segment the current data into smaller windows
window_size = 1000
num_windows = int(len(normalized_data) / window_size)
segments = np.zeros((num_windows, window_size))
for i in range(num_windows):
    start = i * window_size
    end = start + window_size
    segments[i,:] = normalized_data[start:end]

# Extract features from each segment of data
features = np.zeros((num_windows, 4))
for i in range(num_windows):
    features[i,:] = extract_features(segments[i,:])
# Compute the rolling mean and standard deviation of each feature
rolling_mean = pd.DataFrame(features).rolling(window=10).mean().values
rolling_std = pd.DataFrame(features).rolling(window=10).std().values

# Compute the condition indicators for each feature
ci_mean_current = rolling_mean[:,0]
ci_std_current = rolling_std[:,1]
ci_psd_mean = rolling_mean[:,2]
ci_psd_trend = np.polyfit(np.arange(num_windows), rolling_mean[:,2], deg=1)[0]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create a binary label for each data point indicating whether it is normal or anomalous
labels = np.zeros(num_windows)
anomaly_threshold = 3.0  # Threshold for classifying a data point as anomalous
for i in range(num_windows):
    if np.abs(rolling_mean[i,0] - ci_mean_current[i]) > anomaly_threshold * ci_std_current[i]:
        labels[i] = 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy: ", accuracy)

# Load the new current data into a pandas DataFrame
new_data = pd.read_csv('new_motor_current_data.csv')

# Preprocess the data using the same steps as before
filtered_data = signal.filtfilt(b, a, new_data, axis=0)
smoothed_data = np.convolve(filtered_data, np.ones((10,))/10, mode='valid')
normalized_data = (smoothed_data - np.mean(smoothed_data)) / np.std(smoothed_data)

# Segment the data and extract features
num_windows = int(len(normalized_data) / window_size)
segments = np.zeros((num_windows, window_size))
for i in range(num_windows):
    start = i * window_size
    end = start + window_size
    segments[i,:] = normalized_data[start:end]
    
features = np.zeros((num_windows, 4))
for i in range(num_windows):
    features[i,:] = extract_features(segments[i,:])

# Predict the class of each data point using the trained model
y_pred = clf.predict(features)

# Detect anomalies and trigger an alert
anomaly_indices = np.where(y_pred == 1)[0]
if len(anomaly_indices) > 0:
    print("Anomaly detected at indices: ", anomaly_indices)
    # Trigger alert
else:
    print("No anomalies detected.")