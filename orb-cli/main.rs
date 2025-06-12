use orb_cli::{OrbMax, Config};
use image::{ImageReader, Rgba, RgbaImage};
use imageproc::drawing::draw_hollow_circle_mut;
use std::time::Instant;

fn main() {
    // Load grayscale image
    let img = ImageReader::open("lenna.png")
        .expect("Image not found")
        .decode()
        .expect("Decode failed")
        .to_luma8();

    let (w, h) = img.dimensions();
    let orb = OrbMax::new(Config::default(), w as usize, h as usize);

    // Time the full pipeline
    let t0 = Instant::now();
    let (kps, desc) = orb.detect_and_describe(img.as_raw());
    let elapsed = t0.elapsed();

    println!("Time taken: {:.2?}", elapsed);
    println!("Detected {} keypoints", kps.len());
    println!("Generated {} descriptors", desc.len());

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
        .expect("Failed to save output image");
    println!("Saved result image as lenna_keypoints.png");
} 