use super::base::{MatrixSolver, MatrixSolverTrait, Variables};
use crate::num::{Matrix3, Number, Point2, Vector3};

use nalgebra::{Dynamic, MatrixMN, U1, U3, U9};
use nalgebra_lapack::SVD;

pub type HomographySolver<'a> = MatrixSolver<'a, HomographySolverTrait>;

pub struct HomographySolverTrait;

impl MatrixSolverTrait for HomographySolverTrait {
    #[inline]
    fn reverse_transform(mut transform_new: Matrix3) -> Matrix3 {
        let result = transform_new.try_inverse_mut();
        assert!(result);
        transform_new
    }

    fn update_inliers(vars: &mut Variables<Self>) -> f64 {
        // chi-squared value (p=0.05, n=2)
        const CHI_SQ_THRESHOLD: f64 = 5.991;

        let m_new_to_old = &vars.m_new_to_old_in_sac;

        let m_old_to_new = m_new_to_old.try_inverse().unwrap();

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
                    score += CHI_SQ_THRESHOLD - chi_sq_old;
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
                    score += CHI_SQ_THRESHOLD - chi_sq_new;
                }
            }
        }
        score
    }

    fn compute_m_new_to_old(old: &[Point2], new: &[Point2]) -> Matrix3 {
        let num_points = old.len();

        type CoeffMatrix = MatrixMN<Number, Dynamic, U9>;
        let matrix_a = {
            let mut matrix_a = CoeffMatrix::zeros(num_points * 2);

            for i in 0..num_points {
                let old = old[i].to_homogeneous().transpose();
                let new = &new[i];

                let n_row = i * 2;
                matrix_a.index_mut((n_row, 0..3)).fill(0.0);
                matrix_a.index_mut((n_row, 3..6)).copy_from(&-old);
                matrix_a.index_mut((n_row, 6..9)).copy_from(&(new.y * old));

                let n_row = n_row + 1;
                matrix_a.index_mut((n_row, 0..3)).copy_from(&old);
                matrix_a.index_mut((n_row, 3..6)).fill(0.0);
                matrix_a.index_mut((n_row, 6..9)).copy_from(&(-new.x * old));
            }
            matrix_a
        };

        let svd = SVD::new(matrix_a).unwrap();
        Matrix3::from_iterator(svd.vt.index((8, ..)).into_iter().map(|x| *x))
    }

    /// Motion and structure from motion in a piecewise planar environment
    /// (Faugeras et al. in IJPRAI 1988)
    fn decompose(
        m_new_to_old: Matrix3,
        cam_matrix_old: &Matrix3,
        cam_matrix_new: &Matrix3,
    ) -> Option<(Vec<Matrix3>, Vec<Vector3>)> {
        let matrix_a = cam_matrix_new.try_inverse().unwrap() * m_new_to_old * cam_matrix_old;

        // Eq.(7) SVD
        let svd = SVD::new(matrix_a).unwrap();

        let matrix_u = svd.u;
        let lambda = svd.singular_values;
        let matrix_v_t = svd.vt;

        let d1 = lambda[0];
        let d2 = lambda[1];
        let d3 = lambda[2];

        // check rank condition
        if d1 / d2 < 1.0001 || d2 / d3 < 1.0001 {
            return None;
        }

        // intermediate variable in Eq.(8)
        let matrix_s = matrix_u.determinant() * matrix_v_t.determinant();

        // x1 and x3 in Eq.(12)
        let aux_1 = ((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3)).sqrt();
        let aux_3 = ((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3)).sqrt();
        let x1s = [aux_1, aux_1, -aux_1, -aux_1];
        let x3s = [aux_3, -aux_3, aux_3, -aux_3];

        let mut init_rots = vec![];
        let mut init_transes = vec![];

        // when d' > 0

        // Eq.(13)
        let aux_sin_theta = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).sqrt() / ((d1 + d3) * d2);
        let cos_theta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        let aux_sin_thetas = [aux_sin_theta, -aux_sin_theta, -aux_sin_theta, aux_sin_theta];

        for ((&x1, &x3), &aux_sin_theta) in x1s.iter().zip(x3s.iter()).zip(aux_sin_thetas.iter()) {
            // Eq.(13)
            let mut aux_rot = Matrix3::identity();
            aux_rot[(0, 0)] = cos_theta;
            aux_rot[(0, 2)] = -aux_sin_theta;
            aux_rot[(2, 0)] = aux_sin_theta;
            aux_rot[(2, 2)] = cos_theta;

            // Eq.(8)
            let init_rot = matrix_s * matrix_u * aux_rot * matrix_v_t;
            init_rots.push(init_rot);

            // Eq.(14)
            let mut aux_trans = Vector3::new(x1, 0.0, -x3);
            aux_trans *= d1 - d3;

            // Eq.(8)
            let init_trans = matrix_u * aux_trans;
            init_transes.push(init_trans / init_trans.norm());
        }

        // when d' < 0

        // Eq.(13)
        let aux_sin_phi = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).sqrt() / ((d1 - d3) * d2);
        let cos_phi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        let sin_phis = [aux_sin_phi, -aux_sin_phi, -aux_sin_phi, aux_sin_phi];

        for ((&x1, &x3), &sin_phi) in x1s.iter().zip(x3s.iter()).zip(sin_phis.iter()) {
            // Eq.(15)
            let mut aux_rot = Matrix3::identity();
            aux_rot[(0, 0)] = cos_phi;
            aux_rot[(0, 2)] = sin_phi;
            aux_rot[(1, 1)] = -1.0;
            aux_rot[(2, 0)] = sin_phi;
            aux_rot[(2, 2)] = -cos_phi;

            // Eq.(8)
            let init_rot = matrix_s * matrix_u * aux_rot * matrix_v_t;
            init_rots.push(init_rot);
            // Eq.(16)
            let mut aux_trans = Vector3::new(x1, 0.0, x3);
            aux_trans *= d1 + d3;

            // Eq.(8)
            let init_trans = matrix_u * aux_trans;
            init_transes.push(init_trans / init_trans.norm());
        }

        return Some((init_rots, init_transes));
    }
}

impl HomographySolverTrait {
    /// Compute symmetric transfer error
    #[inline]
    fn solve_transfer_error(
        h_21: &Matrix3,
        point_1: &MatrixMN<Number, U3, U1>,
        point_2: &MatrixMN<Number, U3, U1>,
        inv_sigma_sq: f64,
    ) -> f64 {
        // Transform a point in shot 1 to the epipolar line in shot 2,
        // then compute a transfer error (= dot product)
        let mut transformed_point_1 = h_21 * point_1;
        transformed_point_1 = transformed_point_1 / transformed_point_1[2];

        let dist_sq_1 = (point_2 - transformed_point_1).norm_squared();

        // standardization
        dist_sq_1 * inv_sigma_sq
    }
}
