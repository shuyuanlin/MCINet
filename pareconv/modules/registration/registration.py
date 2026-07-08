import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import apply_transform
from pareconv.modules.registration import WeightedProcrustes, solve_local_rotations

from pareconv.modules.registration.pose_estimator import DIS_method


class Dynamic_Inlier_Selection(nn.Module):
    def __init__(
            self,
            k: int,
            acceptance_radius: float,
            confidence_threshold: float = 0.025,
            num_hypotheses: int = 1000,
            num_refinement_steps: int = 5,
            inlier_threshold: float = 0.1,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(Dynamic_Inlier_Selection, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.confidence_threshold = confidence_threshold
        self.num_hypotheses = num_hypotheses
        self.inlier_threshold = inlier_threshold
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        """Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        # correspondences used to vote for hypotheses
        voter_corr_mat = torch.logical_or(torch.gt(src_score_mat, self.confidence_threshold), torch.gt(src_score_mat, self.confidence_threshold))

        # top-k hypotheses used to generate hypotheses
        num_correspondences = min(self.num_hypotheses, mask_mat.sum())
        corr_scores, corr_indices = score_mat.reshape(-1).topk(k=num_correspondences, largest=True)
        # batch_sel_indices = corr_indices // (score_mat.shape[1] * score_mat.shape[2])
        batch_sel_indices = torch.div(corr_indices, (score_mat.shape[1] * score_mat.shape[2]), rounding_mode='floor')
        ref_sel_indices0 = corr_indices % (score_mat.shape[1] * score_mat.shape[2])
        # ref_sel_indices = ref_sel_indices0 // (score_mat.shape[2])
        ref_sel_indices = torch.div(ref_sel_indices0, score_mat.shape[2], rounding_mode='floor')
        src_sel_indices = ref_sel_indices0 % score_mat.shape[1]
        corr_mat = torch.zeros_like(mask_mat, device=mask_mat.device)
        corr_mat[batch_sel_indices, ref_sel_indices, src_sel_indices] = True

        corr_mat = torch.logical_and(corr_mat, mask_mat)
        voter_corr_mat = torch.logical_and(voter_corr_mat, mask_mat)
        batch_indices, row_indices, col_indices = torch.nonzero(voter_corr_mat, as_tuple=True)
        return corr_mat, voter_corr_mat

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def extract_fine_transforms(self, ref_corr_feats, src_corr_feats, ref_corr_points, src_corr_points):
        point_rotations = solve_local_rotations(src_corr_feats, ref_corr_feats)  # B 3 3
        aligned_src_points = torch.einsum('bmn, bn->bm', point_rotations, src_corr_points)
        t = ref_corr_points - aligned_src_points
        transforms = torch.eye(4, device=ref_corr_feats.device).unsqueeze(0).repeat(t.shape[0], 1, 1)
        transforms[:, :3, :3] = point_rotations
        transforms[:, :3, 3] = t
        return transforms

    def dynamic_inlier_selection(self, ref_knn_points,
                                          src_knn_points,
                                          ref_knn_feats,
                                          src_knn_feats,
                                          score_mat,
                                          corr_mat,
                                          voter_corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        ref_corr_feats, src_corr_feats = ref_knn_feats[batch_indices, ref_indices], src_knn_feats[batch_indices, src_indices]

        # pose estimation
        dis_method = DIS_method().to('cuda')
        estimated_transform = dis_method.iterative_pose_estimation(global_src_corr_points.unsqueeze(0), global_ref_corr_points.unsqueeze(0), global_corr_scores)
        estimated_transform = estimated_transform.squeeze(0)
        transformed_src_points = apply_transform(global_src_corr_points, estimated_transform)
        distance = torch.norm(transformed_src_points - global_ref_corr_points, dim=-1)

        valid_mask = distance < self.inlier_threshold
        global_ref_corr_points = global_ref_corr_points[valid_mask]
        global_src_corr_points = global_src_corr_points[valid_mask]
        global_corr_scores = global_corr_scores[valid_mask]
        ref_corr_feats = ref_corr_feats[valid_mask]
        src_corr_feats = src_corr_feats[valid_mask]

        # build verification set
        batch_v_indices, ref_v_indices, src_v_indices = torch.nonzero(voter_corr_mat, as_tuple=True)
        ref_corr_points = ref_knn_points[batch_v_indices, ref_v_indices]
        src_corr_points = src_knn_points[batch_v_indices, src_v_indices]
        corr_scores = score_mat[batch_v_indices, ref_v_indices, src_v_indices]

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform, estimated_transform, ref_corr_feats, src_corr_feats,

    def forward(
            self,
            ref_knn_points,
            src_knn_points,
            re_ref_knn_feats,
            re_src_knn_feats,
            ref_knn_masks,
            src_knn_masks,
            score_mat,

    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (N, K, 3)
            src_knn_points (Tensor): (N, K, 3)
            re_ref_knn_feats (Tensor): (N, K, D, 3)
            re_src_knn_feats (Tensor): (N, K, D, 3)
            ref_knn_masks (BoolTensor): (N, K)
            src_knn_masks (BoolTensor): (N, K)
            score_mat (Tensor): (B, K, K)
        Returns:
            ref_corr_points: (Tensor) (C, 3)
            src_corr_points: (Tensor) (C, 3)
            corr_scores: (Tensor) (C,)
            estimated_transform: (Tensor) (4, 4)
            hypotheses: (Tensor) (N, 4, 4)
            ref_corr_feats: (Tensor) (N, D, 3)
            src_corr_feats: (Tensor) (N, D, 3)
        """

        corr_mat, voter_corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)

        ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, ref_corr_feats, src_corr_feats, \
            = self.dynamic_inlier_selection(
            ref_knn_points,
            src_knn_points,
            re_ref_knn_feats,
            re_src_knn_feats,
            score_mat,
            corr_mat,
            voter_corr_mat
        )
        return ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, ref_corr_feats, src_corr_feats,

