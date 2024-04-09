use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use nanstats::{NaNMean, NaNVar};
use rand::Rng;
use std::iter::repeat_with;

fn bench_nanmean_f32(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("nanmean_f32");
    group.plot_config(plot_config);
    for size in [
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ] {
        group.throughput(Throughput::Bytes((size * 4).try_into().unwrap()));
        let xs: Vec<_> = repeat_with(|| rng.gen::<f32>()).take(size).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(xs.nanmean()));
        });
    }
}

fn bench_nanmean_f64(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("nanmean_f64");
    group.plot_config(plot_config);
    for size in [
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ] {
        group.throughput(Throughput::Bytes((size * 4).try_into().unwrap()));
        let xs: Vec<_> = repeat_with(|| rng.gen::<f64>()).take(size).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(xs.nanmean()));
        });
    }
}

fn bench_nanvar_f32(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("nanvar_f32");
    group.plot_config(plot_config);
    for size in [
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ] {
        group.throughput(Throughput::Bytes((size * 4).try_into().unwrap()));
        let xs: Vec<_> = repeat_with(|| rng.gen::<f32>()).take(size).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(xs.nanvar()));
        });
    }
}

criterion_group!(
    benches,
    bench_nanmean_f32,
    bench_nanmean_f64,
    bench_nanvar_f32
);
criterion_main!(benches);
