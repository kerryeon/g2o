use crate::num::{Matrix3, Point2};

use itertools::Itertools;
use num_traits::Num;

pub fn normalize(points: &[&Point2]) -> (Vec<Point2>, Matrix3) {
    let num_points = points.len() as f64;

    let (mean_x, mean_y) = {
        let points = points.iter().map(|p| (p.x, p.y));
        let (sum_x, sum_y) = sum_tuple2(points);
        (sum_x / num_points, sum_y / num_points)
    };

    let mut normalized = points
        .iter()
        .map(|p| {
            let x = p.x - mean_x;
            let y = p.y - mean_y;
            Point2::new(x, y)
        })
        .collect_vec();

    let (mean_l1_dev_x, mean_l1_dev_y) = {
        let points = normalized.iter().map(|p| (p.x.abs(), p.y.abs()));
        let (sum_x, sum_y) = sum_tuple2(points);
        (sum_x / num_points, sum_y / num_points)
    };

    let mean_l1_dev_x_inv = 1.0 / mean_l1_dev_x;
    let mean_l1_dev_y_inv = 1.0 / mean_l1_dev_y;

    for point in normalized.iter_mut() {
        point.x *= mean_l1_dev_x_inv;
        point.y *= mean_l1_dev_y_inv;
    }

    let transform = Matrix3::new(
        mean_l1_dev_x_inv,
        0.0,
        -mean_x * mean_l1_dev_x_inv,
        0.0,
        mean_l1_dev_y_inv,
        -mean_y * mean_l1_dev_y_inv,
        0.0,
        0.0,
        1.0,
    );

    (normalized, transform)
}

#[inline]
fn sum_tuple2<I, N>(iter: I) -> (N, N)
where
    I: Iterator<Item = (N, N)>,
    N: Num,
{
    iter.fold((N::zero(), N::zero()), |(x1, y1), (x2, y2)| {
        (x1 + x2, y1 + y2)
    })
}
