use orb_cli::{OrbMax, Config};
use image::{ImageReader, Rgba, RgbaImage};
use imageproc::drawing::draw_hollow_circle_mut;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load grayscale image
    let img = ImageReader::open("lenna.png")
        .map_err(|e| format!("Failed to open image: {}", e))?
        .decode()
        .map_err(|e| format!("Failed to decode image: {}", e))?
        .to_luma8();

    let (w, h) = img.dimensions();
    println!("Processing image: {}x{}", w, h);

    // Create ORB detector with error handling
    let orb = match OrbMax::new(Config::default(), w as usize, h as usize) {
        Ok(orb) => orb,
        Err(e) => {
            eprintln!("Failed to create ORB detector: {}", e);
            return Err(e.into());
        }
    };

    println!("ORB detector created successfully");
    println!("Configuration: {:?}", orb.config());

    // Time the full pipeline
    let t0 = Instant::now();
    let (kps, desc) = match orb.detect_and_describe(img.as_raw()) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Failed to detect features: {}", e);
            return Err(e.into());
        }
    };
    let elapsed = t0.elapsed();

    println!("Time taken: {:.2?}", elapsed);
    println!("Detected {} keypoints", kps.len());
    println!("Generated {} descriptors", desc.len());

    if kps.is_empty() {
        println!("Warning: No keypoints detected. Consider adjusting threshold or checking image quality.");
        return Ok(());
    }

    // Convert image to RGBA for drawing
    let mut output: RgbaImage = image::DynamicImage::ImageLuma8(img).into_rgba8();

    // Draw red circles at each keypoint
    for kp in &kps {
        draw_hollow_circle_mut(
            &mut output,
            (kp.x as i32, kp.y as i32),
            3,
            Rgba([255, 0, 0, 255]),
        );
    }

    // Save result
    output
        .save("lenna_keypoints.png")
        .map_err(|e| format!("Failed to save output image: {}", e))?;
    
    println!("Saved result image as lenna_keypoints.png");
    Ok(())
} 