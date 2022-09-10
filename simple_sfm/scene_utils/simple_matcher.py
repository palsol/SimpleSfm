__all__ = ['SimpleMatcher']

import torch
import numpy as np


class SimpleMatcher(object):
    def __init__(self, nn_thresh: float):
        self.nn_thresh = nn_thresh

    def match_two_way(self,
                      desc1: torch.Tensor,
                      desc2: torch.Tensor,
                      nn_thresh: float = None) -> np.array:
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
          desc1 - NxM torch.Tensor of N corresponding M-dimensional descriptors.
          desc2 - NxM torch.Tensor of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        if nn_thresh is None:
            nn_thresh = self.nn_thresh
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')

        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = desc1.t() @ desc2
        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, -1, 1))
        dmat = dmat.cpu().numpy()

        # dmat = np.dot(desc1.T, desc2)

        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores

        return matches

    def match_two_way_cuda(self,
                           desc1: torch.Tensor,
                           desc2: torch.Tensor,
                           nn_thresh: float = None) -> torch.Tensor:
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
          desc1 - NxM torch.Tensor of N corresponding M-dimensional descriptors.
          desc2 - NxM torch.Tensor of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
          matches - 3xL torch.Tensor, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """

        if nn_thresh is None:
            nn_thresh = self.nn_thresh

        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = desc1.t() @ desc2
        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, -1, 1))
        # dmat = dmat.cpu().numpy()
        # # dmat = np.dot(desc1.T, desc2)
        #
        # # Get NN indices and scores.
        idx = torch.argmin(dmat, dim=1)
        scores = dmat[torch.arange(dmat.shape[0]), idx]
        # # Threshold the NN matches.
        keep = scores < nn_thresh
        # # Check if nearest neighbor goes both directions and keep those.
        idx2 = torch.argmin(dmat, dim=0)
        keep_bi = torch.arange(len(idx), device=desc1.device) == idx2[idx]
        keep = torch.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # # Get the surviving point indices.
        m_idx1 = torch.arange(desc1.shape[1], device=desc1.device)[keep]
        m_idx2 = idx
        # # Populate the final 3xN match data structure.
        matches = torch.zeros((3, int(keep.sum())), device=desc1.device)
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores

        return matches
