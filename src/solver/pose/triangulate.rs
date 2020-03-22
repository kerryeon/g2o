use super::super::base::Solver;
use crate::num::{Matrix2, Matrix3, Vector2, Vector3};

pub struct Triangulator<'a> {
    pub old_bearing: &'a Vector3,
    pub new_bearing: &'a Vector3,

    pub rotation_new_to_old: &'a Matrix3,
    pub translation_new_to_old: &'a Vector3,
}

impl<'a> Solver<Vector3> for Triangulator<'a> {
    fn solve(self, _: usize) -> Vector3 {
        let rotation_old_to_new = self.rotation_new_to_old.transpose();

        let translation_old_to_new = -rotation_old_to_new * self.translation_new_to_old;
        let bearing_new_in_old = rotation_old_to_new * self.new_bearing;

        let c = self.old_bearing.dot(&bearing_new_in_old);
        let matrix_a = Matrix2::new(
            self.old_bearing.dot(&self.old_bearing),
            -c,
            c,
            -bearing_new_in_old.dot(&bearing_new_in_old),
        );

        let b = Vector2::new(
            self.old_bearing.dot(&translation_old_to_new),
            bearing_new_in_old.dot(&translation_old_to_new),
        );

        let lambda = matrix_a.try_inverse().unwrap() * b;
        let point_old = lambda[0] * self.old_bearing;
        let point_new = lambda[1] * bearing_new_in_old + translation_old_to_new;
        (point_old + point_new) / 2.0
    }
}
