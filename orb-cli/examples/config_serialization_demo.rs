#[cfg(feature = "serde")]
use orb_fast::DetectorConfig;
use orb_core::Image;
use image::ImageReader;
use std::time::Instant;

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 ORB Configuration Serialization Demo");
    println!("========================================\n");

    // Load test image first to get correct dimensions
    let img_reader = ImageReader::open("lenna.png")?;
    let img = img_reader.decode()?.to_luma8();
    let (w, h) = img.dimensions();
    let width = w as usize;
    let height = h as usize;
    let orb_image: Image = img.as_raw().clone();
    
    println!("📷 Image dimensions: {}x{}", width, height);

    // Demo 1: Create configurations
    println!("📋 Demo 1: Creating Configurations");
    
    let fast_config = DetectorConfig::ultra_fast_preset(width, height)
        .with_metadata("Production Fast", "Optimized for real-time applications");
    
    let quality_config = DetectorConfig::balanced_preset(width, height)
        .with_metadata("Research Quality", "Maximum feature detection quality");
    
    let custom_config = DetectorConfig::new(width, height)
        .with_metadata("Custom Config", "Balanced configuration for specific use case");
    
    println!("   Created 3 configurations:");
    println!("   • {}", fast_config.summary());
    println!("   • {}", quality_config.summary());
    println!("   • {}", custom_config.summary());

    // Demo 2: JSON Serialization
    println!("\n📄 Demo 2: JSON Serialization");
    
    let fast_json = fast_config.to_json()?;
    println!("   Fast config JSON (first 200 chars):");
    println!("   {}", &fast_json[..200.min(fast_json.len())]);
    
    // Save to files
    fast_config.save_json("fast_config.json")?;
    quality_config.save_json("quality_config.json")?;
    custom_config.save_json("custom_config.json")?;
    
    println!("   ✅ Saved 3 JSON configuration files");

    // Demo 3: TOML Serialization
    println!("\n📋 Demo 3: TOML Serialization");
    
    let quality_toml = quality_config.to_toml()?;
    println!("   Quality config TOML (first 300 chars):");
    println!("   {}", &quality_toml[..300.min(quality_toml.len())]);
    
    // Save to files
    fast_config.save_toml("fast_config.toml")?;
    quality_config.save_toml("quality_config.toml")?;
    custom_config.save_toml("custom_config.toml")?;
    
    println!("   ✅ Saved 3 TOML configuration files");

    // Demo 4: Load and Validate Configurations
    println!("\n🔍 Demo 4: Loading and Validation");
    
    let loaded_fast_json = DetectorConfig::load_json("fast_config.json")?;
    let loaded_quality_toml = DetectorConfig::load_toml("quality_config.toml")?;
    
    println!("   Loaded configurations:");
    println!("   • From JSON: {}", loaded_fast_json.summary());
    println!("   • From TOML: {}", loaded_quality_toml.summary());
    
    // Validate configurations
    loaded_fast_json.validate()?;
    loaded_quality_toml.validate()?;
    println!("   ✅ All loaded configurations are valid");

    // Demo 5: Configuration-Based Detection
    println!("\n🎯 Demo 5: Configuration-Based Detection");
    
    println!("   Running detection with different configurations:");
    
    // Test each configuration
    let configs = vec![
        ("Fast (JSON)", loaded_fast_json),
        ("Quality (TOML)", loaded_quality_toml),
    ];
    
    for (name, config) in configs {
        let start = Instant::now();
        let detector = config.to_builder().build()?;
        let keypoints = detector.detect_keypoints(&orb_image)?;
        let elapsed = start.elapsed();
        
        println!("   • {}: {:.2?}, {} keypoints", name, elapsed, keypoints.len());
    }

    // Demo 6: Configuration Comparison
    println!("\n📊 Demo 6: Configuration Comparison");
    
    let configs_to_compare = vec![
        ("Ultra Fast", DetectorConfig::ultra_fast_preset(width, height)),
        ("Balanced", DetectorConfig::balanced_preset(width, height)),
        ("Precision", DetectorConfig::precision_preset(width, height)),
    ];
    
    println!("   Configuration comparison table:");
    println!("   {:<20} {:<8} {:<8} {:<6} {:<8} {:<8}", "Name", "Thresh", "NMS", "Harris", "Adaptive", "CLAHE");
    println!("   {}", "-".repeat(70));
    
    for (name, config) in &configs_to_compare {
        println!("   {:<20} {:<8} {:<8.1} {:<6} {:<8} {:<8}",
            name,
            config.core.threshold,
            config.nms_distance,
            if config.enable_harris_corners { "Yes" } else { "No" },
            if config.enable_adaptive_thresholding { "Yes" } else { "No" },
            if config.enable_clahe_preprocessing { "Yes" } else { "No" }
        );
    }

    // Demo 7: Configuration Templates
    println!("\n📝 Demo 7: Configuration Templates");
    
    // Save configuration templates
    for (name, config) in configs_to_compare {
        let filename = format!("{}_template.json", name.to_lowercase().replace(" ", "_"));
        config.save_json(&filename)?;
        println!("   Saved template: {}", filename);
    }
    
    println!("   ✅ Created configuration templates for easy reuse");

    // Demo 8: Round-trip Testing
    println!("\n🔄 Demo 8: Round-trip Testing");
    
    let original = DetectorConfig::balanced_preset(256, 256)
        .with_metadata("Round-trip Test", "Testing serialization consistency");
    
    // JSON round-trip
    let json_str = original.to_json()?;
    let from_json = DetectorConfig::from_json(&json_str)?;
    
    // TOML round-trip
    let toml_str = original.to_toml()?;
    let from_toml = DetectorConfig::from_toml(&toml_str)?;
    
    println!("   Round-trip test results:");
    println!("   • Original:  {}", original.summary());
    println!("   • From JSON: {}", from_json.summary());
    println!("   • From TOML: {}", from_toml.summary());
    
    // Verify consistency
    assert_eq!(original.width, from_json.width);
    assert_eq!(original.width, from_toml.width);
    assert_eq!(original.core.threshold, from_json.core.threshold);
    assert_eq!(original.core.threshold, from_toml.core.threshold);
    assert_eq!(original.nms_distance, from_json.nms_distance);
    assert_eq!(original.nms_distance, from_toml.nms_distance);
    
    println!("   ✅ Round-trip serialization is consistent");

    println!("\n🎉 Configuration serialization demo completed successfully!");
    println!("📁 Generated files:");
    println!("   • fast_config.json / fast_config.toml");
    println!("   • quality_config.json / quality_config.toml");
    println!("   • custom_config.json / custom_config.toml");
    println!("   • ultra_fast_template.json");
    println!("   • balanced_template.json");
    println!("   • precision_template.json");

    Ok(())
}

#[cfg(not(feature = "serde"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 ORB Configuration Serialization Demo");
    println!("========================================\n");
    println!("❌ This demo requires the 'serde' feature to be enabled.");
    println!("   Run with: cargo run --example config_serialization_demo --features=serde");
    Ok(())
} 