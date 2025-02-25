import numpy as np 
import torch 
torch.manual_seed(0)
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from models.models import EnhancedAttentionLSTM
from utils import load_patient_data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import random
from data_loaders import CHBMITContrastiveDataset_v2


########Determenistic fix############
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)
random.seed(1)

def prepare_patient_split(data_path, all_patients, sampling_freq, n_test_patients=1, window_size_sec=2, test_patients=None):
    """Prepare a single train/test split based on patients"""

    if test_patients is None:
        test_patients = random.sample(all_patients, n_test_patients)
    else: 
        if not isinstance(test_patients, list):
            raise ValueError("test_patients must be a list of patient IDs")

    train_patients = [p for p in all_patients if p not in test_patients]
    
    print(f"Test patients: {test_patients}")
    print(f"Train patients: {train_patients}")
    

    train_windows = []
    train_labels = []
    train_patient_labels = []  # Track patient IDs
    for patient in train_patients:
        windows, labels = load_patient_data(data_path, patient, sampling_freq, window_size_sec)
        if windows is not None:
            train_windows.append(windows)
            train_labels.append(labels)
            train_patient_labels.extend([patient] * len(labels))  # Add patient ID for each window
    

    test_windows = []
    test_labels = []
    test_patient_labels = []  # Track patient IDs
    for patient in test_patients:
        windows, labels = load_patient_data(data_path, patient, sampling_freq, window_size_sec)
        if windows is not None:
            test_windows.append(windows)
            test_labels.append(labels)
            test_patient_labels.extend([patient] * len(labels))  # Add patient ID for each window
    

    train_windows = np.concatenate(train_windows, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_windows = np.concatenate(test_windows, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Channel-wise normalization
    for channel in range(train_windows.shape[2]):  # Iterate over channels
        scaler = StandardScaler()
        scaler = scaler.fit(train_windows[:,:,channel])
        train_windows[:,:,channel] = scaler.transform(train_windows[:,:,channel])
        test_windows[:,:,channel] = scaler.transform(test_windows[:,:,channel])
    
    # Transpose for model input (batch, channels, time)
    train_windows = train_windows.transpose(0, 2, 1)
    test_windows = test_windows.transpose(0, 2, 1)
    
    return (train_windows, train_labels, train_patient_labels, 
            test_windows, test_labels, test_patient_labels)

    
def create_partial_dataset(X, y, percentage):
    """Create a dataset with only a percentage of the data."""
    num_samples = int(len(X) * percentage)
    indices = np.random.choice(len(X), num_samples, replace=False)
    return X[indices], y[indices]

def evaluate_classifiers(train_embeddings, test_embeddings, train_labels_array, test_labels_array):
    results = {}
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(kernel='rbf', probability=True),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, clf in classifiers.items():
        clf.fit(train_embeddings, train_labels_array)
        y_pred = clf.predict(test_embeddings)
        y_prob = clf.predict_proba(test_embeddings)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(test_labels_array, y_pred).ravel()
        results[name] = {
            'accuracy': clf.score(test_embeddings, test_labels_array),
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'auc': auc(*roc_curve(test_labels_array, y_prob)[:2])
        }
        
        print(f"\n{name} Results:")
        for metric, value in results[name].items():
            print(f'{metric.capitalize()}: {value:.4f}')
    
    return results

def main():

    data_path = "./data/processed/CHB-MIT/biclass/"
    n_channels_23 = ["chb01", "chb02", "chb03", "chb04","chb05", "chb06", "chb07", "chb08", "chb09",
                     "chb10", "chb23", "chb24"]
    n_channels_28 = ["chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18",  "chb19", "chb20", "chb21", "chb22"]
    all_patients = n_channels_23 + n_channels_28
    all_results = {patient: {} for patient in all_patients}
    metrics = ['accuracy', 'sensitivity', 'specificity', 'auc']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling_frequency = 128
    hidden_size = 128
    batch_size = 64

    embeddings_dict = {}
    labels_dict = {}

    for test_patient in all_patients:

        (train_windows, train_labels, train_patient_labels, 
        test_windows, test_labels, test_patient_labels) = prepare_patient_split(
            data_path, all_patients, sampling_frequency, n_test_patients=1, test_patients=[test_patient])
        
        print(f"Training data shape: {train_windows.shape}, Labels: {train_labels.shape}")
        print(f"Test data shape: {test_windows.shape}, Labels: {test_labels.shape}")

        model = EnhancedAttentionLSTM(
            input_size=train_windows.shape[2],
            hidden_size=hidden_size,
            n_channels=train_windows.shape[1],
            sfreq=sampling_frequency
            ).to(device)
        
        model_path = f"./models/fixed_2s/ssl_model_{test_patient}.pt"

        try:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        except FileNotFoundError:
            print(f"No saved model found")


        train_dataset = CHBMITContrastiveDataset_v2(train_windows, train_labels)
        test_dataset = CHBMITContrastiveDataset_v2(test_windows, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, drop_last=False)


        model.eval()
        train_embeddings = []
        train_labels_list = []
        test_embeddings = []
        test_labels_list = []

        with torch.no_grad():
            for anchor, _, _, label in train_loader:
                anchor = anchor.to(device)
                embed = model.get_embedding(anchor)
                train_embeddings.append(embed.cpu().numpy())
                train_labels_list.append(label.numpy())
            
            for anchor, _, _, label in test_loader:
                anchor = anchor.to(device)
                embed = model.get_embedding(anchor)
                test_embeddings.append(embed.cpu().numpy())
                test_labels_list.append(label.numpy())
    
        train_embeddings = np.vstack(train_embeddings)
        train_labels_array = np.concatenate(train_labels_list)
        test_embeddings = np.vstack(test_embeddings)
        test_labels_array = np.concatenate(test_labels_list)
     
        all_results[test_patient] = evaluate_classifiers(train_embeddings, test_embeddings, 
                                                   train_labels_array, test_labels_array)
        
        embeddings_dict[test_patient] = test_embeddings
        labels_dict[test_patient] = test_labels_array
    
    print("\nMean Results:")
    for classifier in ['KNN', 'SVM', 'RF']:
        print(f"\n{classifier}:")
        for metric in metrics:
            mean_value = np.mean([all_results[p][classifier][metric] for p in all_patients])
            std_value = np.std([all_results[p][classifier][metric] for p in all_patients])
            print(f'{metric.capitalize()}: {mean_value:.4f} ± {std_value:.4f}')

if __name__ == "__main__":
    main()
        



        



