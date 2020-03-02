use nalgebra::geometry::UnitQuaternion;
use nalgebra::Vector3;
use std::ops::{Mul, MulAssign};

use crate::num::Number;

#[derive(Clone, Debug)]
pub struct SE3Quat {
    pub translation: Vector3<Number>,
    pub rotation: UnitQuaternion<Number>,
}

impl Default for SE3Quat {
    #[inline]
    fn default() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
        }
    }
}

impl<'q> Mul for &'q SE3Quat {
    type Output = SE3Quat;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        result *= rhs;
        return result;
    }
}

impl<'q> MulAssign<&'q Self> for SE3Quat {
    #[inline]
    fn mul_assign(&mut self, rhs: &'q Self) {
        self.translation += self.rotation * rhs.translation;
        self.rotation *= rhs.rotation;
    }
}

impl SE3Quat {
    #[inline]
    pub fn new(translation: Vector3<Number>, rotation: UnitQuaternion<Number>) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        let mut result = self.clone();
        result.inverse_mut();
        return result;
    }

    #[inline]
    pub fn inverse_mut(&mut self) {
        self.translation = self.rotation * (self.translation * -1f64);
        self.rotation.inverse_mut();
    }
}
