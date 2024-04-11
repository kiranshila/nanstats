use pulp::Simd;

pub trait NaNMean<T> {
    fn nanmean(&self) -> T;
}

impl<T> NaNMean<f32> for T
where
    T: AsRef<[f32]>,
{
    fn nanmean(&self) -> f32 {
        nanmean_f32(self.as_ref())
    }
}

impl<T> NaNMean<f64> for T
where
    T: AsRef<[f64]>,
{
    fn nanmean(&self) -> f64 {
        nanmean_f64(self.as_ref())
    }
}

#[pulp::with_simd(nanmean_f32 = pulp::Arch::new())]
#[inline(always)]
pub fn nanmean_f32_with_simd<S: Simd>(simd: S, xs: &[f32]) -> f32 {
    // Constants
    let ones = simd.f32s_splat(1.);

    // Four accumulators to leverage instruction-level parallelism
    let mut count0 = simd.f32s_splat(0.0);
    let mut count1 = simd.f32s_splat(0.0);
    let mut count2 = simd.f32s_splat(0.0);
    let mut count3 = simd.f32s_splat(0.0);

    let mut sum0 = simd.f32s_splat(0.0);
    let mut sum1 = simd.f32s_splat(0.0);
    let mut sum2 = simd.f32s_splat(0.0);
    let mut sum3 = simd.f32s_splat(0.0);

    // Split input chunks
    let (head, tail) = S::f32s_as_simd(xs);
    let (head4, head1) = pulp::as_arrays::<4, _>(head);

    // Deal with the 4 simd chunk, chunks
    for &[x0, x1, x2, x3] in head4 {
        // Create the NaN masks
        let mask0 = simd.f32s_equal(x0, x0);
        let mask1 = simd.f32s_equal(x1, x1);
        let mask2 = simd.f32s_equal(x2, x2);
        let mask3 = simd.f32s_equal(x3, x3);

        // Accumulate counts
        count0 = simd.m32s_select_f32s(mask0, simd.f32s_add(ones, count0), count0);
        count1 = simd.m32s_select_f32s(mask1, simd.f32s_add(ones, count1), count1);
        count2 = simd.m32s_select_f32s(mask2, simd.f32s_add(ones, count2), count2);
        count3 = simd.m32s_select_f32s(mask3, simd.f32s_add(ones, count3), count3);

        // Accumulate masked values
        sum0 = simd.m32s_select_f32s(mask0, simd.f32s_add(x0, sum0), sum0);
        sum1 = simd.m32s_select_f32s(mask1, simd.f32s_add(x1, sum1), sum1);
        sum2 = simd.m32s_select_f32s(mask2, simd.f32s_add(x2, sum2), sum2);
        sum3 = simd.m32s_select_f32s(mask3, simd.f32s_add(x3, sum3), sum3);
    }

    // Then deal with the rest of the chunk
    for &x0 in head1 {
        let mask0 = simd.f32s_equal(x0, x0);
        count0 = simd.m32s_select_f32s(mask0, simd.f32s_add(ones, count0), count0);
        sum0 = simd.m32s_select_f32s(mask0, simd.f32s_add(x0, sum0), sum0);
    }

    // Parallel reduce the sums and counts
    let sum0 = simd.f32s_add(sum0, sum1);
    let sum2 = simd.f32s_add(sum2, sum3);
    let sum0 = simd.f32s_add(sum0, sum2);
    let mut sum = simd.f32s_reduce_sum(sum0);

    let count0 = simd.f32s_add(count0, count1);
    let count2 = simd.f32s_add(count2, count3);
    let count0 = simd.f32s_add(count0, count2);
    let mut count = simd.f32s_reduce_sum(count0);

    tail.iter().for_each(|x| {
        if !x.is_nan() {
            count += 1.0;
            sum += x;
        }
    });

    // Return the mean
    sum / count
}

#[pulp::with_simd(nanmean_f64 = pulp::Arch::new())]
#[inline(always)]
pub fn nanmean_f64_with_simd<S: Simd>(simd: S, xs: &[f64]) -> f64 {
    // Constants
    let ones = simd.f64s_splat(1.);

    // Four accumulators to leverage instruction-level parallelism
    let mut count0 = simd.f64s_splat(0.0);
    let mut count1 = simd.f64s_splat(0.0);
    let mut count2 = simd.f64s_splat(0.0);
    let mut count3 = simd.f64s_splat(0.0);

    let mut sum0 = simd.f64s_splat(0.0);
    let mut sum1 = simd.f64s_splat(0.0);
    let mut sum2 = simd.f64s_splat(0.0);
    let mut sum3 = simd.f64s_splat(0.0);

    // Split input chunks
    let (head, tail) = S::f64s_as_simd(xs);
    let (head4, head1) = pulp::as_arrays::<4, _>(head);

    // Deal with the 4 simd chunk, chunks
    for &[x0, x1, x2, x3] in head4 {
        // Create the NaN masks
        let mask0 = simd.f64s_equal(x0, x0);
        let mask1 = simd.f64s_equal(x1, x1);
        let mask2 = simd.f64s_equal(x2, x2);
        let mask3 = simd.f64s_equal(x3, x3);

        // Accumulate counts
        count0 = simd.m64s_select_f64s(mask0, simd.f64s_add(ones, count0), count0);
        count1 = simd.m64s_select_f64s(mask1, simd.f64s_add(ones, count0), count1);
        count2 = simd.m64s_select_f64s(mask2, simd.f64s_add(ones, count0), count2);
        count3 = simd.m64s_select_f64s(mask3, simd.f64s_add(ones, count0), count3);

        // Accumulate masked values
        sum0 = simd.m64s_select_f64s(mask0, simd.f64s_add(x0, sum0), sum0);
        sum1 = simd.m64s_select_f64s(mask1, simd.f64s_add(x1, sum0), sum1);
        sum2 = simd.m64s_select_f64s(mask2, simd.f64s_add(x2, sum0), sum2);
        sum3 = simd.m64s_select_f64s(mask3, simd.f64s_add(x3, sum0), sum3);
    }

    // Then deal with the rest of the chunk
    for &x0 in head1 {
        let mask0 = simd.f64s_equal(x0, x0);
        count0 = simd.m64s_select_f64s(mask0, simd.f64s_add(ones, count0), count0);
        sum0 = simd.m64s_select_f64s(mask0, simd.f64s_add(x0, sum0), sum0);
    }

    // Parallel reduce the sums and counts
    let sum0 = simd.f64s_add(sum0, sum1);
    let sum2 = simd.f64s_add(sum2, sum3);
    let sum0 = simd.f64s_add(sum0, sum2);
    let mut sum = simd.f64s_reduce_sum(sum0);

    let count0 = simd.f64s_add(count0, count1);
    let count2 = simd.f64s_add(count2, count3);
    let count0 = simd.f64s_add(count0, count2);
    let mut count = simd.f64s_reduce_sum(count0);

    tail.iter().for_each(|x| {
        if !x.is_nan() {
            count += 1.0;
            sum += x;
        }
    });

    // Return the mean
    sum / count
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_f32_nanmean() {
        let mut xs = (0..=2048).map(|x| x as f32).collect::<Vec<_>>();
        xs.push(f32::NAN);
        let mean = xs.nanmean();
        assert_eq!(mean, 1024.);
    }

    #[test]
    fn test_f64_nanmean() {
        let mut xs = (0..=2048).map(|x| x as f64).collect::<Vec<_>>();
        xs.push(f64::NAN);
        let mean = xs.nanmean();
        assert_eq!(mean, 1024.);
    }
}
