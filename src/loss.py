import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal


class NTXentLoss(torch.nn.Module):
    """
    This is the same implementation used in TF-C and TS-TCC. 
    The comparison is between two positive augmentations, the rest of the batch are negative samples.
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, anchor):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    

class FrequencyAwareNTXentLoss(NTXentLoss):
    def __init__(self, device, batch_size, temperature, alpha = 2.0, k_hard=5, fs=128):
        super().__init__(device, batch_size, temperature)
        self.k_hard = k_hard
        self.fs = fs
        self.alpha =  alpha

    def compute_frequency_distances(self, anchor_batch):
        """
        Vectorized computation of frequency band distances
        """
        # Define frequency bands (Hz)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        
        avg_signals = anchor_batch.mean(dim=1).cpu().numpy()
        
        f, psds = signal.welch(avg_signals, fs=self.fs, axis=1)
        band_powers = np.zeros((avg_signals.shape[0], len(bands)))
        for i, (band, (low, high)) in enumerate(bands.items()):
            idx = np.logical_and(f >= low, f <= high)
            # Compute power for all signals in this band
            band_powers[:, i] = np.trapz(psds[:, idx], f[idx], axis=1)
        
        
        band_powers = torch.tensor(band_powers).float().to(self.device)
        band_powers = band_powers / torch.sum(band_powers, dim=1, keepdim=True)
        
        dist_matrix = torch.cdist(band_powers, band_powers)
    
        return dist_matrix
    
    
    def get_intermediate_indices(self, dist_matrix, k, lower_percentile=25, upper_percentile=75):
        """
        Find indices of samples with intermediate distances
        
        Args:
            dist_matrix: pairwise distance matrix
            k: number of negatives to select
            lower_percentile: lower bound percentile
            upper_percentile: upper bound percentile
        """
        
        lower_bound = torch.quantile(dist_matrix, lower_percentile/100.0, dim=1)
        upper_bound = torch.quantile(dist_matrix, upper_percentile/100.0, dim=1)
        
        
        mask = (dist_matrix >= lower_bound.unsqueeze(1)) & (dist_matrix <= upper_bound.unsqueeze(1))
        
        
        dist_matrix_masked = dist_matrix.clone()
        dist_matrix_masked[~mask] = float('-inf')
        
        intermediate_indices = torch.topk(dist_matrix_masked, k=k, dim=1).indices
        
        return intermediate_indices

    def forward(self, zis, zjs, anchors):
        """
        Args:
            zis, zjs: augmented representations from the encoder
            anchors: original signals before augmentation
        """
        # Original NT-Xent computation
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        
       
        freq_dist_matrix = self.compute_frequency_distances(anchors)

        hard_indices = self.get_intermediate_indices(freq_dist_matrix, k=self.k_hard,lower_percentile=25, upper_percentile=75)
        
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        
        hard_negative_mask = torch.zeros_like(negatives)
        for i in range(self.batch_size):
            for idx in hard_indices[i]:
                # Find where this index appears in negatives
                hard_negative_mask[i, idx] = self.alpha
                hard_negative_mask[i + self.batch_size, idx] = self.alpha
        
        # Combine regular and hard negative losses
        logits = torch.cat((positives, negatives * (1 + hard_negative_mask)), dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        
        return loss / (2 * self.batch_size)
        