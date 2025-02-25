from torch.utils.data import Dataset
import torch 
import numpy as np 


class CHBMITContrastiveDataset_v2(Dataset):
    '''
    Dataset class which generates two temporal-spatial augmentations
    '''
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = y
        self.sampling_freq = 128  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor = self.X[idx]
        label = self.y[idx]
        
        # Generate two different augmentations
        aug1 = self.augment_sample(anchor)
        aug2 = self.augment_sample(anchor)

        return anchor, aug1, aug2, label

    def augment_sample(self, sample):
        augmented = sample.clone()
        
        spatial_augmentations = [
            self.apply_channel_dropout,
            self.apply_channel_swap
        ]
        
        temporal_augmentations = [
            self.apply_time_shift,
            self.add_gaussian_noise,
            self.apply_zero_masking
        ]
        
        # Apply one augmentation from each category
        #augmented = np.random.choice(frequency_augmentations)(augmented)
        augmented = np.random.choice(spatial_augmentations)(augmented)
        augmented = np.random.choice(temporal_augmentations)(augmented)
        
        
        return augmented

    def validate_augmentation(self, original, augmented, min_correlation=0.3):
        """Ensure augmentation hasn't destroyed the signal"""
        corr = torch.corrcoef(torch.stack([original.flatten(), augmented.flatten()]))[0,1]
        return corr >= min_correlation

    def add_gaussian_noise(self, sample, std_range=(0.01, 0.05)):
        """Add random Gaussian noise with varying intensity"""
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        noise = torch.randn_like(sample) * std
        return sample + noise

    def apply_time_shift(self, sample, max_shift_ratio=0.15):
        """Apply random time shift"""
        max_shift = int(sample.shape[1] * max_shift_ratio)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(sample, shifts=shift, dims=1)

    def apply_channel_dropout(self, sample, drop_prob=0.1):
        """Randomly drop entire channels"""
        mask = torch.rand(sample.shape[0]) >= drop_prob
        masked_sample = sample.clone()
        masked_sample[~mask] = 0
        return masked_sample

    def apply_channel_swap(self, sample, swap_prob=0.1):
        """Randomly swap adjacent channels"""
        augmented = sample.clone()
        for i in range(sample.shape[0] - 1):
            if torch.rand(1) < swap_prob:
                augmented[i], augmented[i+1] = augmented[i+1].clone(), augmented[i].clone()
        return augmented

    def apply_zero_masking(self, sample, mask_prob_range=(0.1, 0.2)):
        """Randomly mask values to zero"""
        mask_prob = torch.FloatTensor(1).uniform_(*mask_prob_range).item()
        mask = torch.rand_like(sample) < mask_prob
        masked_sample = sample.clone()
        masked_sample[mask] = 0
        return masked_sample

    def apply_amplitude_scaling(self, sample, scale_range=(0.8, 1.2)):
        """Scale the amplitude by a random factor"""
        scale = torch.FloatTensor(1).uniform_(*scale_range)
        return sample * scale
    