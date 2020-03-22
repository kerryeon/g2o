use std::marker::PhantomData;

use super::super::base::Solver;
use super::super::common::normalize;
use crate::num::{Matrix3, Point2, Vector3};

use itertools::Itertools;
use rand::{distributions::Uniform, rngs::ThreadRng, thread_rng, Rng};

mod consts {
    pub const MIN_SET_SIZE: usize = 8;
}

pub trait MatrixSolverTrait: Send + Sync {
    fn reverse_transform(transform_new: Matrix3) -> Matrix3;

    fn update_inliers(vars: &mut Variables<Self>) -> f64
    where
        Self: Sized;

    fn compute_m_new_to_old(old: &[Point2], new: &[Point2]) -> Matrix3;

    fn decompose(
        m_new_to_old: Matrix3,
        cam_matrix_old: &Matrix3,
        cam_matrix_new: &Matrix3,
    ) -> Option<(Vec<Matrix3>, Vec<Vector3>)>;
}

pub struct MatrixSolver<'a, T>
where
    T: MatrixSolverTrait,
{
    pub solver: PhantomData<T>,

    pub old: &'a [&'a Point2],
    pub new: &'a [&'a Point2],

    pub matches: &'a [(usize, usize)],

    pub sigma: f64,
}

pub struct MatrixResult {
    pub is_inlier_match: Vec<bool>,

    pub best_score: f64,
    pub best_m_new_to_old: Matrix3,
}

pub struct Variables<'a, T>
where
    T: MatrixSolverTrait,
{
    solver: PhantomData<T>,
    pub(super) old: &'a [&'a Point2],
    pub(super) new: &'a [&'a Point2],

    normalized_old: Vec<Point2>,
    normalized_new: Vec<Point2>,

    transform_old: &'a Matrix3,
    transform_new_inv: &'a Matrix3,

    pub(super) matches: &'a [(usize, usize)],

    pub(super) sigma: f64,

    rng: ThreadRng,

    // RANSAC variables
    best_score: Option<f64>,
    best_m_new_to_old: Option<Matrix3>,
    is_inlier_match: Option<Vec<bool>>,

    // minimum set of keypoint matches
    min_set_keypts_old: Vec<Point2>,
    min_set_keypts_new: Vec<Point2>,

    // shared variables in RANSAC loop
    pub(super) m_new_to_old_in_sac: Matrix3,

    // inlier/outlier flags
    pub(super) is_inlier_match_in_sac: Vec<bool>,
}

impl<'a, T> Solver<Option<MatrixResult>> for MatrixSolver<'a, T>
where
    T: MatrixSolverTrait,
{
    fn solve(self, repeat: usize) -> Option<MatrixResult> {
        // 0. Normalize keypoint coordinates
        let (normalized_old, transform_old) = normalize(&self.old);
        let (normalized_new, transform_new) = normalize(&self.new);

        let transform_new_inv = T::reverse_transform(transform_new);

        // 1. Prepare for RANSAC
        let mut vars = {
            if self.matches.len() < consts::MIN_SET_SIZE {
                return None;
            }

            Variables::new(
                self.old,
                self.new,
                normalized_old,
                normalized_new,
                &transform_old,
                &transform_new_inv,
                self.matches,
                self.sigma,
            )
        };

        // 2. RANSAC loop
        for _ in 0..repeat {
            vars.update();
        }
        if !vars.is_valid() {
            return None;
        }

        // 3. Recompute a matrix only with the inlier matches

        let (inlier_normalized_keypts_old, inlier_normalized_keypts_new) =
            vars.filter_inlier_points();

        let normalized_m_new_to_old = T::compute_m_new_to_old(
            inlier_normalized_keypts_old.as_slice(),
            inlier_normalized_keypts_new.as_slice(),
        );
        vars.m_new_to_old_in_sac = transform_new_inv * normalized_m_new_to_old * transform_old;
        let best_score = T::update_inliers(&mut vars);

        Some(MatrixResult {
            is_inlier_match: vars.is_inlier_match_in_sac,
            best_score,
            best_m_new_to_old: vars.m_new_to_old_in_sac,
        })
    }
}

/// main implementation
impl<'a, T> Variables<'a, T>
where
    T: MatrixSolverTrait,
{
    #[inline]
    fn new(
        old: &'a [&'a Point2],
        new: &'a [&'a Point2],
        normalized_old: Vec<Point2>,
        normalized_new: Vec<Point2>,
        transform_old: &'a Matrix3,
        transform_new_inv: &'a Matrix3,
        matches: &'a [(usize, usize)],
        sigma: f64,
    ) -> Self {
        Self {
            solver: Default::default(),

            old,
            new,

            normalized_old,
            normalized_new,

            transform_old,
            transform_new_inv,
            matches,

            sigma,

            rng: thread_rng(),

            // RANSAC variables
            best_score: None,
            best_m_new_to_old: None,
            is_inlier_match: None,

            // minimum set of keypoint matches
            min_set_keypts_old: vec![Point2::new(0.0, 0.0); consts::MIN_SET_SIZE],
            min_set_keypts_new: vec![Point2::new(0.0, 0.0); consts::MIN_SET_SIZE],

            // shared variables in RANSAC loop
            m_new_to_old_in_sac: Matrix3::zeros(),

            // inlier/outlier flags
            is_inlier_match_in_sac: vec![false; matches.len()],
        }
    }

    #[inline]
    fn update(&mut self) {
        // 1. Create a minimum set
        for (i, idx) in self
            .rng
            .sample_iter(Uniform::new(0, self.matches.len()))
            .take(consts::MIN_SET_SIZE)
            .enumerate()
        {
            let (old, new) = self.matches[idx];
            self.min_set_keypts_old[i] = self.normalized_old[old];
            self.min_set_keypts_new[i] = self.normalized_new[new];
        }

        // 2. Compute a matrix
        let normalized_m_new_to_old =
            T::compute_m_new_to_old(&self.min_set_keypts_old, &self.min_set_keypts_new);
        self.m_new_to_old_in_sac =
            self.transform_new_inv * normalized_m_new_to_old * self.transform_old;

        // 3. Check inliers and compute a score
        let score_in_sac = T::update_inliers(self);

        // 4. Update the best model
        if self.best_score.unwrap_or(0.0) < score_in_sac {
            self.best_score.replace(score_in_sac);
            self.best_m_new_to_old.replace(self.m_new_to_old_in_sac);
            self.is_inlier_match
                .replace(self.is_inlier_match_in_sac.clone());
        }
    }

    #[inline]
    fn is_valid(&self) -> bool {
        match self.is_inlier_match.as_ref() {
            Some(is_inlier_match) => {
                is_inlier_match.iter().filter(|x| **x).count() >= consts::MIN_SET_SIZE
            }
            None => false,
        }
    }

    #[inline]
    fn filter_inlier_points(&self) -> (Vec<Point2>, Vec<Point2>) {
        let is_inlier_match = self.is_inlier_match.as_ref().unwrap();
        let points = self
            .matches
            .iter()
            .enumerate()
            .filter(|(i, _)| is_inlier_match[*i])
            .map(|(_, (old, new))| (self.normalized_old[*old], self.normalized_new[*new]))
            .collect_vec();
        let points_old = points.iter().map(|(p, _)| p.clone()).collect();
        let points_new = points.iter().map(|(_, p)| p.clone()).collect();
        (points_old, points_new)
    }
}
