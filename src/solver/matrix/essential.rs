use super::base::{MatrixSolverTrait, Variables};
use crate::num::{Matrix3, Point2, Vector3};

use nalgebra_lapack::SVD;

// pub type EssentialSolver<'a> = MatrixSolver<'a, EssentialSolverTrait>;

pub struct EssentialSolverTrait;

impl MatrixSolverTrait for EssentialSolverTrait {
    #[inline]
    fn reverse_transform(_: Matrix3) -> Matrix3 {
        unimplemented!();
    }

    fn update_inliers(_: &mut Variables<Self>) -> f64 {
        unimplemented!();
    }

    fn compute_m_new_to_old(_: &[Point2], _: &[Point2]) -> Matrix3 {
        unimplemented!();
    }

    /// https://en.wikipedia.org/wiki/Essential_matrix#Determining_R_and_t_from_E
    fn decompose(
        m_new_to_old: Matrix3,
        _: &Matrix3,
        _: &Matrix3,
    ) -> Option<(Vec<Matrix3>, Vec<Vector3>)> {
        let svd = SVD::new(m_new_to_old).unwrap();
        let matrix_u = svd.u;
        let matrix_v_t = svd.vt;

        let mut trans: Vector3 = matrix_u.index((.., 2)).into();
        trans.normalize_mut();

        let mut matrix_w = Matrix3::zeros();
        matrix_w[(0, 1)] = -1.0;
        matrix_w[(1, 0)] = 1.0;
        matrix_w[(2, 2)] = 1.0;

        let mut rot_1 = matrix_u * matrix_w * matrix_v_t;
        if rot_1.determinant() < 0.0 {
            rot_1 *= -1.0;
        }

        let mut rot_2 = matrix_u * matrix_w.transpose() * matrix_v_t;
        if rot_2.determinant() < 0.0 {
            rot_2 *= -1.0;
        }

        let init_rots = vec![rot_1, rot_1, rot_2, rot_2];
        let init_transes = vec![trans, -trans, trans, -trans];
        Some((init_rots, init_transes))
    }
}
