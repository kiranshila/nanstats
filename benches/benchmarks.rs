use aligned_vec::{avec, AVec};
use divan::{black_box, counter::BytesCount};
use nanstats::{NaNMean, NaNVar};

mod mean {
    use super::*;

    #[divan::bench(consts = [16384, 32768, 65536])]
    fn f32<const N: usize>(bencher: divan::Bencher) {
        bencher
            .counter(BytesCount::new(N * 4))
            .with_inputs(|| -> AVec<f32> { avec![0f32; N] })
            .input_counter(|x: &AVec<f32>| BytesCount::of_slice(x))
            .bench_refs(|x| black_box(x.nanmean()))
    }

    #[divan::bench(consts = [16384, 32768, 65536])]
    fn f64<const N: usize>(bencher: divan::Bencher) {
        bencher
            .counter(BytesCount::new(N * 4))
            .with_inputs(|| -> AVec<f64> { avec![0f64; N] })
            .input_counter(|x: &AVec<f64>| BytesCount::of_slice(x))
            .bench_refs(|x| black_box(x.nanmean()))
    }
}

mod var {
    use super::*;

    #[divan::bench(consts = [16384, 32768, 65536])]
    fn f32<const N: usize>(bencher: divan::Bencher) {
        bencher
            .counter(BytesCount::new(N * 4))
            .with_inputs(|| -> AVec<f32> { avec![0f32; N] })
            .input_counter(|x: &AVec<f32>| BytesCount::of_slice(x))
            .bench_refs(|x| black_box(x.nanvar()))
    }

    #[divan::bench(consts = [16384, 32768, 65536])]
    fn f64<const N: usize>(bencher: divan::Bencher) {
        bencher
            .counter(BytesCount::new(N * 4))
            .with_inputs(|| -> AVec<f64> { avec![0f64; N] })
            .input_counter(|x: &AVec<f64>| BytesCount::of_slice(x))
            .bench_refs(|x| black_box(x.nanvar()))
    }
}

fn main() {
    // Run registered benchmarks.
    divan::main();
}
