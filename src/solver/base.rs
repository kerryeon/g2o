pub trait Solver<R>: Send + Sync
where
    R: Send + Sync,
{
    const MAX_ITER: usize = 0;

    fn solve(self, repeat: usize) -> R;

    #[inline]
    fn solve_to_end(self) -> R
    where
        Self: Sized,
    {
        self.solve(Self::MAX_ITER)
    }
}
