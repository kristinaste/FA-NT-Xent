import numpy as np
import os 
import datetime

def balance_dataset(windows, labels, ratio=2.0):
    """
    Same as in Helsinki 
    """
    class_1_indices = np.where(labels == 1)[0]
    class_0_indices = np.where(labels == 0)[0]
    
    n_class_1 = len(class_1_indices)
    n_class_0_keep = int(n_class_1 * ratio)
    
    class_0_indices_keep = np.random.choice(class_0_indices, size=n_class_0_keep, replace=False)
    
    keep_indices = np.concatenate([class_1_indices, class_0_indices_keep])
    np.random.shuffle(keep_indices)
    
    return windows[keep_indices], labels[keep_indices] 


def balance_dataset_prodrome(windows, labels, ratio=2):
    """Balance dataset for three classes"""
    ictal_indices = np.where(labels == 2)[0]
    preictal_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    
    n_ictal = len(ictal_indices)
    n_samples = n_ictal * ratio
    
    # Randomly sample from normal 
    normal_indices = np.random.choice(normal_indices, n_samples, replace=False)
    #preictal_indices = np.random.choice(preictal_indices, n_samples, replace=False)
    
    # Combine indices, all preictal 
    balanced_indices = np.concatenate([normal_indices, preictal_indices, ictal_indices])
    
    return windows[balanced_indices], labels[balanced_indices]



def create_windows(data, labels, window_size_seconds=4, sampling_freq=128, overlap_seconds=0):
    """
    Create windows from EEG data with configurable overlap
    
    Args:
        data: Input EEG data array
        labels: Input labels array
        window_size_seconds: Size of each window in seconds
        sampling_freq: Sampling frequency of the data
        overlap_seconds: Overlap between consecutive windows in seconds
        
    Returns:
        Tuple of (windowed_data, windowed_labels)
    """
    samples_per_window = window_size_seconds * sampling_freq
    
    if overlap_seconds >= window_size_seconds:
        raise ValueError("Overlap duration must be less than window size")
    
    # Calculate step size based on overlap
    overlap_samples = int(overlap_seconds * sampling_freq)
    step = samples_per_window - overlap_samples
    
    windowed_data = []
    windowed_labels = []
    labels = labels.astype(int)
    
    # Create windows with overlap
    for i in range(0, len(data) - samples_per_window + 1, step):
        window = data[i:i+samples_per_window]
        # Use majority voting for window label
        label = np.bincount(labels[i:i+samples_per_window]).argmax()
        windowed_data.append(window)
        windowed_labels.append(label)
    
    return np.array(windowed_data), np.array(windowed_labels)



def save_results(all_results, params, timestamp=None):
    """Save experiment results and parameters to file"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/experiment_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        # Write parameters
        f.write("=== Experiment Parameters ===\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Sampling Frequency: {params['sampling_frequency']} Hz\n")
        f.write(f"Hidden Size: {params['hidden_size']}\n")
        f.write(f"Number of Epochs: {params['num_epochs']}\n")
        f.write(f"Batch Size: {params['batch_size']}\n")
        f.write(f"Temperature: {params['temperature']}\n")
        f.write(f"Alpha (k-hard negative weight): {params['alpha']}\n")
        f.write(f"Windows size: {params['window_size_sec']}\n")
        f.write("\n")
        
        # Write results for each patient
        f.write("=== Results by Patient ===\n")
        for patient, results in all_results.items():
            f.write(f"\nPatient: {patient}\n")
            for percentage, metrics in results.items():
                f.write(f"\n{percentage*100}% Training Data:\n")
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Sensitivity: {metrics['sensitivity']:.2f}%\n")
                f.write(f"Specificity: {metrics['specificity']:.2f}%\n")
                f.write(f"AUC: {metrics['auc']:.2f}%\n")
        
        # Write summary statistics
        f.write("\n=== Overall Results Summary ===\n")
        for percentage in set(p for r in all_results.values() for p in r.keys()):
            accuracies = [results[percentage]['accuracy'] for results in all_results.values()]
            sensitivities = [results[percentage]['sensitivity'] for results in all_results.values()]
            specificities = [results[percentage]['specificity'] for results in all_results.values()]
            aucs = [results[percentage]['auc'] for results in all_results.values()]
            
            f.write(f"\n{percentage*100}% Training Data:\n")
            f.write(f"Mean Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%\n")
            f.write(f"Mean sensitivity: {np.mean(sensitivities):.2f}% ± {np.std(sensitivities):.2f}%\n")
            f.write(f"Mean specificity: {np.mean(specificities):.2f}% ± {np.std(specificities):.2f}%\n")
            f.write(f"Mean AUC: {np.mean(aucs):.2f}% ± {np.std(aucs):.2f}%\n")
    
    print(f"Results saved to {filename}")


def load_patient_data(data_path, patient, sampling_freq, window_size_sec=2):
    """Load and process data for a single patient"""
    data_file = os.path.join(data_path, f"{patient}/{patient}_data.npy")
    label_file = os.path.join(data_path, f"{patient}/{patient}_labels.npy")
    
    if os.path.exists(data_file) and os.path.exists(label_file):
        data = np.load(data_file)
        labels = np.load(label_file)
        
        windows, windows_labels = create_windows(data, labels, 
                                              window_size_seconds=window_size_sec, 
                                              sampling_freq=sampling_freq, overlap_seconds=0)
        
        windows, windows_labels = balance_dataset(windows, windows_labels, ratio=1)
        return windows, windows_labels
    return None, None

