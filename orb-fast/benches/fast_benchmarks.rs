use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use orb_core::{Image, OrbConfig, FastVariant};
use orb_fast::FastDetector;
use orb_fast::DetectorBuilder;

/// Create benchmark image with realistic corner patterns
fn create_benchmark_image(width: usize, height: usize, complexity: &str) -> Image {
    let mut img = vec![128; width * height];
    
    match complexity {
        "simple" => {
            // Simple corner pattern
            let cx = width / 2;
            let cy = height / 2;
            for dy in -4..=4 {
                for dx in -4..=4 {
                    let x = (cx as i32 + dx) as usize;
                    let y = (cy as i32 + dy) as usize;
                    if x < width && y < height {
                        if dx.abs() <= 2 && dy.abs() <= 2 {
                            img[y * width + x] = 255;
                        }
                    }
                }
            }
        },
        "complex" => {
            // Multiple corners with varying intensities
            let corners = vec![(width/4, height/4), (3*width/4, height/4), 
                              (width/4, 3*height/4), (3*width/4, 3*height/4),
                              (width/2, height/2)];
            
            for (i, &(cx, cy)) in corners.iter().enumerate() {
                let intensity = 50 + (i * 40) as u8;
                for dy in -6..=6 {
                    for dx in -6..=6 {
                        let x = (cx as i32 + dx) as usize;
                        let y = (cy as i32 + dy) as usize;
                        if x < width && y < height {
                            if dx.abs() <= 3 && dy.abs() <= 3 {
                                img[y * width + x] = intensity + 100;
                            }
                        }
                    }
                }
            }
        },
        "realistic" => {
            // Simulate more realistic image with noise and gradients
            for y in 0..height {
                for x in 0..width {
                    let gradient = ((x as f32 / width as f32) * 50.0) as u8;
                    let noise = ((x + y) % 7) as u8;
                    img[y * width + x] = 100 + gradient + noise;
                }
            }
            
            // Add some corner-like structures
            for i in 0..20 {
                let cx = (i * width / 20) % width;
                let cy = (i * height / 20) % height;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let x = (cx as i32 + dx) as usize;
                        let y = (cy as i32 + dy) as usize;
                        if x < width && y < height {
                            img[y * width + x] = if (dx + dy) % 2 == 0 { 50 } else { 200 };
                        }
                    }
                }
            }
        },
        _ => {}
    }
    
    img
}

fn create_test_config() -> OrbConfig {
    OrbConfig {
        threshold: 20,
        patch_size: 15,
        n_threads: 1, // Single-threaded for consistent benchmarks
        fast_variant: FastVariant::Fast9,
        nms_radius: 3.0,
    }
}

/// Benchmark full detection pipeline
fn bench_full_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_detection");
    
    let sizes = vec![
        (64, 64),
        (128, 128), 
        (256, 256),
        (512, 512),
        (567, 567), // Our standard test size
    ];
    
    let complexities = vec!["simple", "complex", "realistic"];
    
    for &(width, height) in &sizes {
        for complexity in &complexities {
            let detector = FastDetector::new(create_test_config(), width, height).unwrap();
            let img = create_benchmark_image(width, height, complexity);
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", width, height), complexity),
                &(detector, img),
                |b, (detector, img)| {
                    b.iter(|| {
                        black_box(detector.detect_keypoints(black_box(img)).unwrap())
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark individual pipeline stages
fn bench_pipeline_stages(c: &mut Criterion) {
    let width = 256;
    let height = 256;
    let detector = FastDetector::new(create_test_config(), width, height).unwrap();
    let img = create_benchmark_image(width, height, "realistic");
    
    let mut group = c.benchmark_group("pipeline_stages");
    
    // Benchmark raw keypoint detection with response
    group.bench_function("detect_keypoints_with_response", |b| {
        b.iter(|| {
            black_box(detector.detect_keypoints_with_response(black_box(&img)).unwrap())
        })
    });
    
    // Get some keypoints for downstream benchmarks
    let scored_keypoints = detector.detect_keypoints_with_response(&img).unwrap();
    
    // Benchmark non-maximum suppression
    group.bench_function("non_maximum_suppression", |b| {
        b.iter(|| {
            black_box(detector.non_maximum_suppression(black_box(&scored_keypoints), 3.0))
        })
    });
    
    // Benchmark subpixel refinement
    if !scored_keypoints.is_empty() {
        let keypoint = scored_keypoints[0].keypoint;
        group.bench_function("subpixel_refinement", |b| {
            b.iter(|| {
                black_box(detector.refine_keypoint_subpixel(black_box(&img), black_box(keypoint)).unwrap())
            })
        });
    }
    
    group.finish();
}

/// Benchmark Harris corner response computation
fn bench_harris_response(c: &mut Criterion) {
    let width = 256;
    let height = 256;
    let detector = FastDetector::new(create_test_config(), width, height).unwrap();
    let img = create_benchmark_image(width, height, "realistic");
    
    let mut group = c.benchmark_group("harris_response");
    
    // Benchmark single Harris computation
    group.bench_function("single_point", |b| {
        b.iter(|| {
            black_box(detector.compute_harris_response(black_box(&img), 128, 128))
        })
    });
    
    // Benchmark Harris for multiple points
    let points: Vec<(usize, usize)> = (0..100)
        .map(|i| (50 + (i % 10) * 15, 50 + (i / 10) * 15))
        .collect();
    
    group.bench_function("100_points", |b| {
        b.iter(|| {
            for &(x, y) in black_box(&points) {
                black_box(detector.compute_harris_response(black_box(&img), x, y));
            }
        })
    });
    
    group.finish();
}

/// Benchmark orientation computation
fn bench_orientation(c: &mut Criterion) {
    let width = 256;
    let height = 256;
    let detector = FastDetector::new(create_test_config(), width, height).unwrap();
    let img = create_benchmark_image(width, height, "realistic");
    
    let mut group = c.benchmark_group("orientation");
    
    // Single orientation computation
    group.bench_function("single_point", |b| {
        b.iter(|| {
            black_box(detector.compute_orientation(black_box(&img), 128.0, 128.0).unwrap())
        })
    });
    
    // Multiple orientation computations
    let points: Vec<(f32, f32)> = (0..100)
        .map(|i| (50.0 + (i % 10) as f32 * 15.5, 50.0 + (i / 10) as f32 * 15.5))
        .collect();
    
    group.bench_function("100_points", |b| {
        b.iter(|| {
            for &(x, y) in black_box(&points) {
                black_box(detector.compute_orientation(black_box(&img), x, y).unwrap());
            }
        })
    });
    
    group.finish();
}

/// Benchmark multi-scale detection components
fn bench_multiscale(c: &mut Criterion) {
    let width = 256;
    let height = 256;
    let detector = FastDetector::new(create_test_config(), width, height).unwrap();
    let img = create_benchmark_image(width, height, "realistic");
    
    let mut group = c.benchmark_group("multiscale");
    
    // Benchmark image pyramid construction
    group.bench_function("build_pyramid", |b| {
        b.iter(|| {
            black_box(orb_fast::pyramid::ImagePyramid::build_image_pyramid(black_box(&img), width, height, detector.get_scale_levels()).unwrap())
        })
    });
    
    // Benchmark single scale level detection
    let scale_levels = detector.get_scale_levels();
    if scale_levels.len() > 1 {
        let pyramid = orb_fast::pyramid::ImagePyramid::build_image_pyramid(&img, width, height, scale_levels).unwrap();
        group.bench_function("single_scale_detection", |b| {
            b.iter(|| {
                black_box(detector.detect_keypoints_at_scale(black_box(&pyramid[1]), black_box(&scale_levels[1])).unwrap())
            })
        });
    }
    
    group.finish();
}

/// Benchmark adaptive thresholding
fn bench_adaptive_threshold(c: &mut Criterion) {
    let width = 256;
    let height = 256;
    let detector = FastDetector::new(create_test_config(), width, height).unwrap();
    
    let mut group = c.benchmark_group("adaptive_threshold");
    
    // Different image types
    let images = vec![
        ("uniform", vec![128; width * height]),
        ("gradient", {
            let mut img = vec![0; width * height];
            for y in 0..height {
                for x in 0..width {
                    img[y * width + x] = ((x + y) * 255 / (width + height)) as u8;
                }
            }
            img
        }),
        ("noisy", create_benchmark_image(width, height, "realistic")),
    ];
    
    for (name, img) in images {
        group.bench_with_input(
            BenchmarkId::new("threshold_map", name),
            &img,
            |b, img| {
                b.iter(|| {
                    black_box(detector.get_adaptive_threshold_map(black_box(img)).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Memory allocation benchmarks
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    let sizes = vec![
        (64, 64),
        (256, 256),
        (512, 512),
    ];
    
    for &(width, height) in &sizes {
        // Benchmark detector creation
        group.bench_with_input(
            BenchmarkId::new("detector_creation", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    black_box(FastDetector::new(create_test_config(), w, h).unwrap())
                })
            },
        );
        
        // Benchmark image creation
        group.bench_with_input(
            BenchmarkId::new("image_creation", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    black_box(create_benchmark_image(w, h, "realistic"))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark CLAHE preprocessing
#[cfg(feature = "clahe-preprocessing")]
fn bench_clahe_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("clahe_preprocessing");
    
    let sizes = vec![
        (128, 128),
        (256, 256),
        (512, 512),
    ];
    
    for &(width, height) in &sizes {
        let detector = FastDetector::new(create_test_config(), width, height).unwrap();
        let img = create_benchmark_image(width, height, "realistic");
        
        // Benchmark CLAHE preprocessing alone
        group.bench_function(&format!("clahe_only/{}x{}", width, height), |b| {
            b.iter(|| {
                black_box(detector.apply_clahe_preprocessing(black_box(&img)).unwrap())
            })
        });
        
        // Benchmark full detection with CLAHE
        group.bench_function(&format!("detection_with_clahe/{}x{}", width, height), |b| {
            b.iter(|| {
                black_box(detector.detect_keypoints_with_response(black_box(&img)).unwrap())
            })
        });
    }
    
    group.finish();
}

fn benchmark_detector_builder_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("DetectorBuilder Presets");
    
    // Test image sizes
    let sizes = [(256, 256), (512, 512), (1024, 1024)];
    
    for &(width, height) in &sizes {
        let img = create_benchmark_image(width, height, "realistic");
        let size_name = format!("{}x{}", width, height);
        
        // Ultra Fast preset benchmark
        group.bench_function(&format!("ultra_fast_preset_{}", size_name), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .preset_ultra_fast()
                    .threshold(25)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
        
        // Balanced preset benchmark
        group.bench_function(&format!("balanced_preset_{}", size_name), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .preset_balanced()
                    .threshold(20)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
        
        // Precision preset benchmark
        group.bench_function(&format!("precision_preset_{}", size_name), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .preset_precision()
                    .threshold(15)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
        
        // Custom configuration benchmark
        group.bench_function(&format!("custom_config_{}", size_name), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .threshold(18)
                    .patch_size(21)
                    .nms_radius(4.5)
                    .subpixel_refinement(true)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
    }
    
    group.finish();
}

fn benchmark_detector_builder_api_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("DetectorBuilder API Overhead");
    
    let (width, height) = (512, 512);
    let img = create_benchmark_image(width, height, "realistic");
    let config = create_test_config();
    
    // Direct FastDetector creation (baseline)
    group.bench_function("direct_fastdetector_creation", |b| {
        b.iter(|| {
            let detector = FastDetector::new(config.clone(), width, height).unwrap();
            detector.detect_keypoints(&img).unwrap()
        })
    });
    
    // DetectorBuilder creation
    group.bench_function("detector_builder_creation", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(config.threshold)
                .patch_size(config.patch_size)
                .threads(config.n_threads)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    // Builder reuse (creating builder once, using multiple times)
    group.bench_function("detector_builder_reuse", |b| {
        let configured = DetectorBuilder::new(width, height)
            .threshold(config.threshold)
            .patch_size(config.patch_size)
            .threads(config.n_threads)
            .build()
            .unwrap();
        
        b.iter(|| {
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_detector_builder_feature_flags(c: &mut Criterion) {
    let mut group = c.benchmark_group("DetectorBuilder Feature Flags");
    
    let (width, height) = (512, 512);
    let img = create_benchmark_image(width, height, "realistic");
    
    // Minimal features
    group.bench_function("minimal_features", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(25)
                .simd(false)
                .optimized_layout(false)
                .harris_corners(false)
                .adaptive_thresholding(false)
                .clahe_preprocessing(false)
                .subpixel_refinement(false)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    // SIMD only
    group.bench_function("simd_only", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(25)
                .simd(true)
                .optimized_layout(false)
                .harris_corners(false)
                .adaptive_thresholding(false)
                .clahe_preprocessing(false)
                .subpixel_refinement(false)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    // Optimized layout only
    group.bench_function("optimized_layout_only", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(25)
                .simd(false)
                .optimized_layout(true)
                .harris_corners(false)
                .adaptive_thresholding(false)
                .clahe_preprocessing(false)
                .subpixel_refinement(false)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    // All features enabled
    group.bench_function("all_features", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(20)
                .simd(true)
                .optimized_layout(true)
                .harris_corners(true)
                .adaptive_thresholding(true)
                .clahe_preprocessing(true)
                .subpixel_refinement(true)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_detector_builder_configuration_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("DetectorBuilder Configuration Impact");
    
    let (width, height) = (512, 512);
    let img = create_benchmark_image(width, height, "realistic");
    
    // Different threshold values
    for threshold in [10, 15, 20, 25, 30] {
        group.bench_function(&format!("threshold_{}", threshold), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .threshold(threshold)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
    }
    
    // Different NMS distances
    for nms_distance in [2.0, 3.0, 4.0, 5.0, 6.0] {
        group.bench_function(&format!("nms_distance_{:.1}", nms_distance), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .threshold(20)
                    .nms_radius(nms_distance)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
    }
    
    // Different patch sizes
    for patch_size in [11, 15, 19, 23, 31] {
        group.bench_function(&format!("patch_size_{}", patch_size), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .threshold(20)
                    .patch_size(patch_size)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
    }
    
    group.finish();
}

fn benchmark_fast_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("FAST Variants");
    
    let (width, height) = (512, 512);
    let img = create_benchmark_image(width, height, "realistic");
    
    // Test different FAST variants
    let variants = [
        ("FAST-6", FastVariant::custom(6)),
        ("FAST-9", FastVariant::Fast9),
        ("FAST-12", FastVariant::Fast12),
        ("FAST-16", FastVariant::Fast16),
        ("FAST-10", FastVariant::custom(10)),
        ("FAST-14", FastVariant::custom(14)),
    ];
    
    for (name, variant) in variants {
        group.bench_function(name, |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .threshold(20)
                    .fast_variant(variant)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
    }
    
    group.finish();
}

fn benchmark_nms_radius_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("NMS Radius Impact");
    
    let (width, height) = (512, 512);
    let img = create_benchmark_image(width, height, "realistic");
    
    // Test different NMS radius values
    for radius in [1.0, 2.0, 3.0, 5.0, 8.0] {
        group.bench_function(&format!("radius_{:.1}", radius), |b| {
            b.iter(|| {
                let configured = DetectorBuilder::new(width, height)
                    .threshold(20)
                    .nms_radius(radius)
                    .build()
                    .unwrap();
                configured.detect_keypoints(&img).unwrap()
            })
        });
    }
    
    group.finish();
}

fn benchmark_enhanced_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("Enhanced Features");
    
    let (width, height) = (512, 512);
    let img = create_benchmark_image(width, height, "realistic");
    
    // Baseline - minimal features
    group.bench_function("baseline", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(20)
                .fast_variant(FastVariant::Fast9)
                .harris_corners(false)
                .adaptive_thresholding(false)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    // With Harris corners
    group.bench_function("with_harris", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(20)
                .fast_variant(FastVariant::Fast9)
                .harris_corners(true)
                .adaptive_thresholding(false)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    // With adaptive thresholding
    group.bench_function("with_adaptive", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(20)
                .fast_variant(FastVariant::Fast9)
                .harris_corners(false)
                .adaptive_thresholding(true)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    // With all enhanced features
    group.bench_function("all_enhanced", |b| {
        b.iter(|| {
            let configured = DetectorBuilder::new(width, height)
                .threshold(20)
                .fast_variant(FastVariant::custom(10))
                .harris_corners(true)
                .adaptive_thresholding(true)
                .subpixel_refinement(true)
                .build()
                .unwrap();
            configured.detect_keypoints(&img).unwrap()
        })
    });
    
    group.finish();
}

#[cfg(feature = "clahe-preprocessing")]
criterion_group!(
    benches,
    bench_full_detection,
    bench_pipeline_stages,
    bench_harris_response,
    bench_orientation,
    bench_multiscale,
    bench_adaptive_threshold,
    bench_memory_patterns,
    bench_clahe_preprocessing,
    benchmark_detector_builder_presets,
    benchmark_detector_builder_api_overhead,
    benchmark_detector_builder_feature_flags,
    benchmark_detector_builder_configuration_impact,
    benchmark_fast_variants,
    benchmark_nms_radius_impact,
    benchmark_enhanced_features
);

#[cfg(not(feature = "clahe-preprocessing"))]
criterion_group!(
    benches,
    bench_full_detection,
    bench_pipeline_stages,
    bench_harris_response,
    bench_orientation,
    bench_multiscale,
    bench_adaptive_threshold,
    bench_memory_patterns,
    benchmark_detector_builder_presets,
    benchmark_detector_builder_api_overhead,
    benchmark_detector_builder_feature_flags,
    benchmark_detector_builder_configuration_impact,
    benchmark_fast_variants,
    benchmark_nms_radius_impact,
    benchmark_enhanced_features
);

criterion_main!(benches); 