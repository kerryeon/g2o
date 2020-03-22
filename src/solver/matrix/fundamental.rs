use super::base::{MatrixSolver, MatrixSolverTrait, Variables};
use super::essential::EssentialSolverTrait;
use crate::num::{Matrix3, Number, Point2, Vector3};

use nalgebra::{Dynamic, MatrixMN, U1, U3, U9};
use nalgebra_lapack::SVD;

pub type FundamentalSolver<'a> = MatrixSolver<'a, FundamentalSolverTrait>;

pub struct FundamentalSolverTrait;

impl MatrixSolverTrait for FundamentalSolverTrait {
    #[inline]
    fn reverse_transform(mut transform_new: Matrix3) -> Matrix3 {
        transform_new.transpose_mut();
        transform_new
    }

    fn update_inliers(vars: &mut Variables<Self>) -> f64 {
        // chi-squared value (p=0.05, n=1)
        const CHI_SQ_THRESHOLD: f64 = 3.841;
        // chi-squared value (p=0.05, n=2)
        const SCORE_THRESHOLD: f64 = 5.991;

        let m_new_to_old = &vars.m_new_to_old_in_sac;

        let m_old_to_new = m_new_to_old.transpose();

        let inv_sigma_sq = 1.0 / (vars.sigma * vars.sigma);

        let mut score = 0.0;
        for i in 0..vars.matches.len() {
            let (old, new) = vars.matches[i];
            let is_inlier_match = &mut vars.is_inlier_match_in_sac[i];

            // 1. Transform to homogeneous coordinates

            let point_old = vars.old[old].to_homogeneous();
            let point_new = vars.new[new].to_homogeneous();

            // 2. Compute symmetric transfer error

            {
                let chi_sq_old =
                    Self::solve_transfer_error(&m_new_to_old, &point_old, &point_new, inv_sigma_sq);

                // if a match is inlier, accumulate the score
                if CHI_SQ_THRESHOLD < chi_sq_old {
                    *is_inlier_match = false;
                    continue;
                } else {
                    *is_inlier_match = true;
                    score += SCORE_THRESHOLD - chi_sq_old;
                }
            }

            {
                let chi_sq_new =
                    Self::solve_transfer_error(&m_old_to_new, &point_new, &point_old, inv_sigma_sq);

                // if a match is inlier, accumulate the score
                if CHI_SQ_THRESHOLD < chi_sq_new {
                    *is_inlier_match = false;
                    continue;
                } else {
                    *is_inlier_match = true;
                    score += SCORE_THRESHOLD - chi_sq_new;
                }
            }
        }
        score
    }

    fn compute_m_new_to_old(old: &[Point2], new: &[Point2]) -> Matrix3 {
        let num_points = old.len();

        type CoeffMatrix = MatrixMN<Number, Dynamic, U9>;
        let matrix_a = {
            let mut matrix_a = CoeffMatrix::zeros(num_points);

            for i in 0..num_points {
                let old = old[i].to_homogeneous().transpose();
                let new = &new[i];

                let n_row = i;
                matrix_a.index_mut((n_row, 0..3)).copy_from(&(new.x * old));
                matrix_a.index_mut((n_row, 3..6)).copy_from(&(new.y * old));
                matrix_a.index_mut((n_row, 6..9)).copy_from(&old);
            }
            matrix_a
        };

        let init_svd = SVD::new(matrix_a).unwrap();
        let init_f_new_to_old: Matrix3 =
            Matrix3::from_iterator(init_svd.vt.index((8, ..)).into_iter().map(|x| *x));

        let svd = SVD::new(init_f_new_to_old).unwrap();

        let matrix_u = svd.u;
        let matrix_v_t = svd.vt;
        let mut lambda = svd.singular_values;

        lambda[2] = 0.0;

        matrix_u * Matrix3::from_diagonal(&lambda) * matrix_v_t
    }

    fn decompose(
        m_new_to_old: Matrix3,
        cam_matrix_old: &Matrix3,
        cam_matrix_new: &Matrix3,
    ) -> Option<(Vec<Matrix3>, Vec<Vector3>)> {
        let e_new_to_old = cam_matrix_new.transpose() * m_new_to_old * cam_matrix_old;
        EssentialSolverTrait::decompose(e_new_to_old, cam_matrix_old, cam_matrix_new)
    }
}

impl FundamentalSolverTrait {
    /// Compute symmetric transfer error
    #[inline]
    fn solve_transfer_error(
        f_21: &Matrix3,
        point_1: &MatrixMN<Number, U3, U1>,
        point_2: &MatrixMN<Number, U3, U1>,
        inv_sigma_sq: f64,
    ) -> f64 {
        // Transform a point in shot 1 to the epipolar line in shot 2,
        // then compute a transfer error (= dot product)
        let epiline_in_2 = f_21 * point_1;

        let residual_in_2 = epiline_in_2.dot(&point_2);
        let dist_sq_2 =
            residual_in_2 * residual_in_2 / epiline_in_2.index((0..2, 0)).norm_squared();

        // standardization
        dist_sq_2 * inv_sigma_sq
    }
}
