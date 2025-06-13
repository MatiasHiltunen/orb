use orb_fast::{DetectorBuilder, FastResult};
use orb_core::Image;
use image::{ImageReader, Rgba, RgbaImage};
use imageproc::drawing::draw_hollow_circle_mut;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 ORB DetectorBuilder API Demo");
    println!("=================================\n");

    // Load grayscale image
    let img_reader = ImageReader::open("lenna.png")
        .map_err(|e| format!("Failed to open image: {}", e))?;
    let img = img_reader
        .decode()
        .map_err(|e| format!("Failed to decode image: {}", e))?
        .to_luma8();

    let (w, h) = img.dimensions();
    let width = w as usize;
    let height = h as usize;
    println!("📷 Processing image: {}x{}", width, height);

    // Convert to our Image format
    let orb_image: Image = img.as_raw().clone();

    // Demo 1: Fast preset (minimal features, maximum speed)
    println!("\n🚀 Demo 1: Fast Preset");
    run_detection_demo(
        DetectorBuilder::new(width, height)
            .preset_fast()
            .threshold(25),
        &orb_image,
        "fast"
    )?;

    // Demo 2: Quality preset (all quality features enabled)
    println!("\n✨ Demo 2: Quality Preset");
    run_detection_demo(
        DetectorBuilder::new(width, height)
            .preset_quality()
            .threshold(20),
        &orb_image,
        "quality"
    )?;

    // Demo 3: Illumination robust preset
    println!("\n🌟 Demo 3: Illumination Robust Preset");
    run_detection_demo(
        DetectorBuilder::new(width, height)
            .preset_illumination_robust()
            .threshold(15),
        &orb_image,
        "illumination_robust"
    )?;

    // Demo 4: Custom configuration
    println!("\n⚙️  Demo 4: Custom Configuration");
    run_detection_demo(
        DetectorBuilder::new(width, height)
            .threshold(18)
            .patch_size(21)
            .threads(4)
            .nms_distance(4.5)
            .subpixel_refinement(true)
            .simd(true)
            .optimized_layout(true),
        &orb_image,
        "custom"
    )?;

    // Demo 5: Performance comparison
    println!("\n⏱️  Demo 5: Performance Comparison");
    performance_comparison(&orb_image, width, height)?;

    println!("\n🎉 All demos completed successfully!");
    println!("Check the generated images: lenna_keypoints_*.png");

    Ok(())
}

fn run_detection_demo(
    builder: DetectorBuilder,
    img: &Image,
    name: &str,
) -> FastResult<()> {
    // Print configuration summary
    println!("   Config: {}", builder.summary());
    
    // Build the detector
    let configured = builder.build()?;
    
    // Time the detection
    let start = Instant::now();
    let keypoints = configured.detect_keypoints(img)?;
    let elapsed = start.elapsed();
    
    println!("   ⏱️  Time: {:.2?}", elapsed);
    println!("   🎯 Detected {} keypoints", keypoints.len());
    
    // Generate keypoint density metric
    let area = configured.dimensions().0 * configured.dimensions().1;
    let density = keypoints.len() as f32 / area as f32 * 10000.0; // per 10k pixels
    println!("   📊 Density: {:.2} keypoints per 10k pixels", density);
    
    // Visualize and save
    if let Err(e) = save_keypoint_visualization(img, &keypoints, &format!("lenna_keypoints_{}.png", name), configured.dimensions()) {
        println!("   ⚠️  Warning: Failed to save visualization: {}", e);
    } else {
        println!("   💾 Saved: lenna_keypoints_{}.png", name);
    }
    
    println!();
    Ok(())
}

fn performance_comparison(img: &Image, width: usize, height: usize) -> Result<(), Box<dyn std::error::Error>> {
    let configs = vec![
        ("Minimal", DetectorBuilder::new(width, height)
            .threshold(30)
            .nms_distance(2.0)
            .subpixel_refinement(false)),
        ("Balanced", DetectorBuilder::new(width, height)
            .threshold(20)
            .nms_distance(4.0)
            .subpixel_refinement(true)),
        ("Maximum", DetectorBuilder::new(width, height)
            .preset_quality()
            .threshold(15)
            .nms_distance(5.0)),
    ];

    println!("   Comparing different configurations:");
    println!("   {:<12} {:<10} {:<10} {:<15}", "Config", "Time", "Keypoints", "Quality Score");
    println!("   {}", "-".repeat(50));

    for (name, builder) in configs {
        let configured = builder.build()?;
        
        // Run detection multiple times for more accurate timing
        let mut total_time = std::time::Duration::ZERO;
        let mut keypoints = Vec::new();
        const RUNS: usize = 3;
        
        for _ in 0..RUNS {
            let start = Instant::now();
            keypoints = configured.detect_keypoints(img)?;
            total_time += start.elapsed();
        }
        
        let avg_time = total_time / RUNS as u32;
        
        // Calculate a simple quality score (response variance)
        let scored_keypoints = configured.detect_keypoints_with_response(img)?;
        let quality_score = if !scored_keypoints.is_empty() {
            let mean_response: f32 = scored_keypoints.iter().map(|sk| sk.response).sum::<f32>() / scored_keypoints.len() as f32;
            let variance: f32 = scored_keypoints.iter()
                .map(|sk| (sk.response - mean_response).powi(2))
                .sum::<f32>() / scored_keypoints.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };
        
        println!("   {:<12} {:<10.2?} {:<10} {:<15.2}", 
                name, avg_time, keypoints.len(), quality_score);
    }
    
    Ok(())
}

fn save_keypoint_visualization(
    img: &Image,
    keypoints: &[orb_core::Keypoint],
    filename: &str,
    (width, height): (usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert to image format for visualization
    let luma_img = image::GrayImage::from_raw(width as u32, height as u32, img.clone())
        .ok_or("Failed to create image from raw data")?;
    
    // Convert to RGBA for drawing
    let mut output: RgbaImage = image::DynamicImage::ImageLuma8(luma_img).into_rgba8();

    // Draw circles at each keypoint with size based on response/scale
    for kp in keypoints {
        let radius = 3; // Fixed radius for simplicity
        let color = if keypoints.len() < 500 {
            Rgba([255, 0, 0, 255]) // Red for sparse
        } else if keypoints.len() < 1500 {
            Rgba([0, 255, 0, 255]) // Green for medium
        } else {
            Rgba([0, 0, 255, 255]) // Blue for dense
        };
        
        draw_hollow_circle_mut(
            &mut output,
            (kp.x.round() as i32, kp.y.round() as i32),
            radius,
            color,
        );
    }

    // Save the result
    output.save(filename)?;
    Ok(())
} 