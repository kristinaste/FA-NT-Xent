import numpy as np 
import pandas as pd
import torch 
torch.manual_seed(0)
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from loss import NTXentLoss, FrequencyAwareNTXentLoss
from models.models import EnhancedAttentionLSTM
from utils import balance_dataset, create_windows, save_results
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os
import random
from data_loaders import CHBMITContrastiveDataset_v2

########Determenistic fix############
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)
random.seed(1)


#######General Workflow########
# First we need to ensure the performance of finetuning
# 
# Split data by test and train patients
# 100% of train patients are used in the unlabelled training 
#
# The test set is strictly used for leave-one-out validation
# The model is trained in self-supervised manner with full 100% of training unlabelled data 
# and then the layers are frozen and the finetuning stage happens. 
# The test is validated on kNN and SVM, and fine-tuned MLP in supervised manner. 


######Utility functions#######
def train_ssl(model, train_loader, device, num_epochs, optimizer, criterion, test_patient):
    """Train the model on unlabelled dataset
    The final model is saved with the name of the test patient to reproduce validation
    """
    save_path = f"./models/fixed_2s/ssl_model_{test_patient}.pt"
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for anchor, aug1, aug2, _ in train_loader:
            anchor, aug1, aug2 = anchor.to(device), aug1.to(device), aug2.to(device)
            
            optimizer.zero_grad()
            z1 = model(aug1)
            z2 = model(aug2)
            
            loss = criterion(z1, z2, anchor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save model state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'loss': avg_loss,
        'test_patient': test_patient
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

    return model

def load_model_for_patient(model, test_patient):
    """Load model saved for specific test patient"""
    model_path = f"./models/fixed_2s/ssl_model_{test_patient}.pt"
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model for test patient {test_patient}")
        return model
    except FileNotFoundError:
        print(f"No saved model found for test patient {test_patient}")
        return None

def freeze_encoder(model):
    """Freeze all layers except the projection head"""
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Remove the projection head 
    model.projection = None

    return model

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

class FineTuningDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class FineTuningHead(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x) 

def finetune_model(base_model, train_windows, train_labels, device, percentage=1.0, 
                  num_epochs=20, batch_size=64):
    """Fine-tune the model with a percentage of labeled data."""
    partial_X, partial_y = create_partial_dataset(train_windows, train_labels, percentage)

    # Print class distribution
    unique, counts = np.unique(partial_y, return_counts=True)
    print(f"Class distribution in training data: {dict(zip(unique, counts))}")
    

    finetune_dataset = FineTuningDataset(partial_X, partial_y)
    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, 
                               shuffle=True, drop_last=True)

    classifier = FineTuningHead(base_model.hidden_size).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in finetune_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                embeddings = base_model.get_embedding(inputs)
            
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(finetune_loader)
        accuracy = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return classifier


def create_balanced_test_dataset(test_windows, test_labels, ratio=1):
    """Create balanced test dataset with specified ratio of non-seizure:seizure samples"""
    seizure_idx = np.where(test_labels == 1)[0]
    nonseizure_idx = np.where(test_labels == 0)[0]
    
    n_seizure = len(seizure_idx)
    n_nonseizure = int(n_seizure * ratio)  # 2:1 ratio
    
    selected_nonseizure_idx = np.random.choice(nonseizure_idx, n_nonseizure, replace=False)
    selected_idx = np.concatenate([selected_nonseizure_idx, seizure_idx])
    np.random.shuffle(selected_idx)
    
    return test_windows[selected_idx], test_labels[selected_idx]



def evaluate_model(base_model, classifier, test_loader, device, balance_test=True):
    """Evaluate the fine-tuned model on test data."""
    base_model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_embeddings = []
    all_labels = []
    predictions = []
    probabilities = []

    all_data = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            all_data.extend(inputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # Balance test set if requested
    if balance_test:
        all_data, all_labels = create_balanced_test_dataset(all_data, all_labels)
        
        # Create new balanced test dataset and loader
        balanced_dataset = FineTuningDataset(all_data, all_labels)
        test_loader = DataLoader(balanced_dataset, batch_size=test_loader.batch_size, 
                                shuffle=False)
        
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = base_model.get_embedding(inputs)
            outputs = classifier(embeddings)
            
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_embeddings.append(embeddings.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
            probabilities.append(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    accuracy = 100 * correct / total
    all_embeddings = np.vstack(all_embeddings)
    test_labels = np.concatenate(test_labels)
    predictions = np.concatenate(predictions)
    probabilities = np.concatenate(probabilities)
    
    TP = np.sum((predictions == 1) & (test_labels == 1))
    TN = np.sum((predictions == 0) & (test_labels == 0))
    FP = np.sum((predictions == 1) & (test_labels == 0))
    FN = np.sum((predictions == 0) & (test_labels == 1))

    print(TP, TN, FP, FN)

    sensitivity = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
    specificity = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0
    accuracy = ((TP + TN) / len(test_labels)) * 100
    auc = roc_auc_score(test_labels, probabilities) * 100
    
    results = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'embeddings': all_embeddings,
        'labels': test_labels
    }
    
    return results

def check_model_exists(test_patient):
    """Check if model exists for given test patient"""
    model_path = f"./models/fixed_2s/ssl_model_{test_patient}.pt"
    return os.path.exists(model_path)

######Main execution loop#####

def main():

    data_path = "./data/processed/CHB-MIT/biclass/"
    n_channels_23 = ["chb01", "chb02", "chb03", "chb04","chb05", "chb06", "chb07", "chb08", "chb09",
                     "chb10", "chb23", "chb24"]
    n_channels_28 = ["chb11", "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18",  "chb19", "chb20", "chb21", "chb22"]
    all_patients = n_channels_23 + n_channels_28

    #Parametes
    sampling_frequency = 128
    hidden_size = 128
    num_epochs = 2         
    batch_size = 64
    temperature = 0.7
    alpha = 3.0 #k-hard negative weight 
    window_size_sec = 2
    train_self_supervised = True #Set False when the self-supervised trained model is saved
    
    fine_tuning_perc = [1, 0.75] 
    ablation = False #Set true to use vanilla NT-Xent loss for ablation purpose

    #We need this to save the results
    all_results = {}
    params = {
        'sampling_frequency': sampling_frequency,
        'hidden_size': hidden_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'temperature': temperature,
        'alpha': alpha,
        'window_size_sec': window_size_sec,
        'experiment_desc': ablation
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_patient in all_patients:

        print(f"\nLeave-one-out validation for test patient: {test_patient}")
        # Prepare data split
        (train_windows, train_labels, _, 
        test_windows, test_labels, _) = prepare_patient_split(
            data_path, all_patients, sampling_frequency, n_test_patients=1, window_size_sec=window_size_sec, test_patients=[test_patient])

        
        print(f"Training data shape: {train_windows.shape}, Labels: {train_labels.shape}")
        print(f"Test data shape: {test_windows.shape}, Labels: {test_labels.shape}")


        train_dataset = CHBMITContrastiveDataset_v2(train_windows, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, drop_last=True)
    

        model = EnhancedAttentionLSTM(
            input_size=train_windows.shape[2],
            hidden_size=hidden_size,
            n_channels=train_windows.shape[1],
            sfreq=sampling_frequency
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if ablation: 
            criterion = NTXentLoss(device, batch_size=batch_size, temperature=temperature)
        else: 
            criterion = FrequencyAwareNTXentLoss(device, batch_size=batch_size, temperature=temperature, alpha=alpha, k_hard=20,  fs = sampling_frequency)
        

        if train_self_supervised and not check_model_exists(test_patient): 
            model = train_ssl(model, train_loader, device, num_epochs, optimizer, criterion, test_patient) 

        else:
            try:
                loaded_model = load_model_for_patient(model, test_patient)
                print(f"Loaded pre-trained model for patient {test_patient} successfully")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print(f"Please train the model first or provide a valid model path for {test_patient}")
                return
            model = loaded_model
            
        model = freeze_encoder(model)

        #it is the same test dataset for all classifiers 
        test_dataset_eval = FineTuningDataset(test_windows, test_labels)
        test_loader_eval = DataLoader(test_dataset_eval, batch_size=batch_size, shuffle=False)

        patient_results = {}
        for percentage in fine_tuning_perc: 
            classifier = finetune_model(model, train_windows, train_labels, device, percentage=percentage, num_epochs=20)

            results = evaluate_model(model, classifier, test_loader_eval, device)

            patient_results[percentage] = {
                'accuracy': results['accuracy'],
                'sensitivity': results['sensitivity'],
                'specificity': results['specificity'],
                'auc': results['auc']
            }
            
            print(f"\nResults for {test_patient} when finetuning on {percentage*100}% of training data:")
            print(f"MLP Classifier:")
            print(f"  Accuracy: {results['accuracy']:.2f}%")
            print(f"  Sensitivity: {results['sensitivity']:.2f}%")
            print(f"  Specificity: {results['specificity']:.2f}%")
            print(f"  AUC: {results['auc']:.2f}%")

        all_results[test_patient] = patient_results

    # Print summary statistics
    print("\nOverall Results Summary:")
    for percentage in fine_tuning_perc:
        accuracies = [results[percentage]['accuracy'] 
                     for results in all_results.values()]
        
        print(f"\n{percentage*100}% Training Data:")
        print(f"Mean Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")

    save_results(all_results, params)

if __name__ == "__main__":
    main()