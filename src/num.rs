pub type Number = f64;

pub type Point2 = nalgebra::Point2<Number>;
pub type Point3 = nalgebra::Point3<Number>;

pub type Vector2 = nalgebra::Vector2<Number>;
pub type Vector3 = nalgebra::Vector3<Number>;

pub type Matrix2 = nalgebra::Matrix2<Number>;
pub type Matrix3 = nalgebra::Matrix3<Number>;

pub type SE3Quat = nalgebra::Isometry3<Number>;

#[test]
fn test_se3_update() {
    use super::SE3Quat;
    use nalgebra::{Quaternion, Translation3, UnitQuaternion};

    let a = Translation3::new(-5.3, 1.8, -9.6);
    let b = UnitQuaternion::try_new(Quaternion::new(1.0, 2.0, 3.0, 1.0), 3.0).unwrap();
    let c = Translation3::new(1.5, 3.2, 7.6);

    let x = SE3Quat::from_parts(a, b);
    let y = SE3Quat::from_parts(c, b);
    let z = SE3Quat::identity();

    println!("{:?}", y * x.inverse());
    println!("{:?}", z * x.inverse());
}
