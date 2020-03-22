use crate::num::{Matrix3, Vector3};

pub struct PoseResult {
    pub rotation_old_to_new: Matrix3,
    pub translations_old_to_new: Vector3,

    pub triangulated_points: Vec<Vector3>,
    pub is_triangulated: Vec<bool>,
}
