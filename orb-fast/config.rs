use orb_core::{Image, Keypoint, OrbConfig};
use crate::error::{FastError, FastResult};
use crate::types::{ScoredKeypoint, ScaleLevel};
use crate::detector::FastDetector;

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

/// Complete detector configuration with all settings
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DetectorConfig {
    /// Core ORB configuration
    pub core: OrbConfig,
    /// Image dimensions
    pub width: usize,
    pub height: usize,
    /// Feature flags
    pub enable_simd: bool,
    pub enable_optimized_layout: bool,
    pub enable_harris_corners: bool,
    pub enable_adaptive_thresholding: bool,
    pub enable_clahe_preprocessing: bool,
    /// Performance tuning
    pub nms_distance: f32,
    pub subpixel_refinement: bool,
    /// Metadata
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub name: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub description: Option<String>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub version: Option<String>,
}

impl DetectorConfig {
    /// Create new configuration with default settings
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            core: OrbConfig {
                threshold: 20,
                patch_size: 31,
                n_threads: 1,
            },
            width,
            height,
            enable_simd: cfg!(feature = "simd"),
            enable_optimized_layout: cfg!(feature = "optimized-memory-layout"),
            enable_harris_corners: false,
            enable_adaptive_thresholding: false,
            enable_clahe_preprocessing: false,
            nms_distance: 3.0,
            subpixel_refinement: true,
            name: None,
            description: None,
            version: None,
        }
    }

    /// Fast preset optimized for speed
    pub fn fast_preset(width: usize, height: usize) -> Self {
        Self {
            core: OrbConfig {
                threshold: 30,
                patch_size: 15,
                n_threads: num_cpus::get(),
            },
            width,
            height,
            enable_simd: true,
            enable_optimized_layout: true,
            enable_harris_corners: false,
            enable_adaptive_thresholding: false,
            enable_clahe_preprocessing: false,
            nms_distance: 5.0,
            subpixel_refinement: false,
            name: Some("Fast".to_string()),
            description: Some("Optimized for maximum speed with minimal features".to_string()),
            version: Some("1.0".to_string()),
        }
    }

    /// Quality preset optimized for detection quality
    pub fn quality_preset(width: usize, height: usize) -> Self {
        Self {
            core: OrbConfig {
                threshold: 15,
                patch_size: 31,
                n_threads: num_cpus::get(),
            },
            width,
            height,
            enable_simd: true,
            enable_optimized_layout: true,
            enable_harris_corners: true,
            enable_adaptive_thresholding: true,
            enable_clahe_preprocessing: false,
            nms_distance: 2.0,
            subpixel_refinement: true,
            name: Some("Quality".to_string()),
            description: Some("Balanced quality and performance with advanced features".to_string()),
            version: Some("1.0".to_string()),
        }
    }

    /// Illumination robust preset for challenging lighting
    pub fn illumination_robust_preset(width: usize, height: usize) -> Self {
        Self {
            core: OrbConfig {
                threshold: 12,
                patch_size: 31,
                n_threads: num_cpus::get(),
            },
            width,
            height,
            enable_simd: true,
            enable_optimized_layout: true,
            enable_harris_corners: true,
            enable_adaptive_thresholding: true,
            enable_clahe_preprocessing: true,
            nms_distance: 2.0,
            subpixel_refinement: true,
            name: Some("Illumination Robust".to_string()),
            description: Some("Maximum robustness for challenging illumination conditions".to_string()),
            version: Some("1.0".to_string()),
        }
    }

    /// Add metadata to configuration
    pub fn with_metadata(mut self, name: &str, description: &str) -> Self {
        self.name = Some(name.to_string());
        self.description = Some(description.to_string());
        self.version = Some("1.0".to_string());
        self
    }

    /// Convert to DetectorBuilder for further customization
    pub fn to_builder(self) -> DetectorBuilder {
        DetectorBuilder::from_config(self)
    }

    /// Generate human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "DetectorConfig: {}x{}, threshold={}, features=[SIMD:{}, OptLayout:{}, Harris:{}, Adaptive:{}, CLAHE:{}]",
            self.width, self.height, self.core.threshold,
            self.enable_simd, self.enable_optimized_layout, self.enable_harris_corners,
            self.enable_adaptive_thresholding, self.enable_clahe_preprocessing
        )
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> FastResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(FastError::InvalidImageSize { width: self.width, height: self.height });
        }
        if self.core.threshold == 0 || self.core.threshold > 127 {
            return Err(FastError::InvalidThreshold(self.core.threshold));
        }
        Ok(())
    }

    /// Save configuration to JSON file
    #[cfg(feature = "serde")]
    pub fn save_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load configuration from JSON file
    #[cfg(feature = "serde")]
    pub fn load_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to TOML file
    #[cfg(feature = "serde")]
    pub fn save_toml<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let toml = toml::to_string_pretty(self)?;
        std::fs::write(path, toml)?;
        Ok(())
    }

    /// Load configuration from TOML file
    #[cfg(feature = "serde")]
    pub fn load_toml<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Serialize to JSON string
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    /// Serialize to TOML string
    #[cfg(feature = "serde")]
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Deserialize from TOML string
    #[cfg(feature = "serde")]
    pub fn from_toml(toml_str: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: Self = toml::from_str(toml_str)?;
        config.validate()?;
        Ok(config)
    }
}

/// Fluent API builder for detector configuration
pub struct DetectorBuilder {
    config: OrbConfig,
    width: usize,
    height: usize,
    // Feature flags
    enable_simd: bool,
    enable_optimized_layout: bool,
    enable_harris_corners: bool,
    enable_adaptive_thresholding: bool,
    enable_clahe_preprocessing: bool,
    // Performance tuning
    nms_distance: f32,
    subpixel_refinement: bool,
}

impl DetectorBuilder {
    /// Create new builder with default settings
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            config: OrbConfig {
                threshold: 20,
                patch_size: 31,
                n_threads: 1,
            },
            width,
            height,
            enable_simd: cfg!(feature = "simd"),
            enable_optimized_layout: cfg!(feature = "optimized-memory-layout"),
            enable_harris_corners: false,
            enable_adaptive_thresholding: false,
            enable_clahe_preprocessing: false,
            nms_distance: 3.0,
            subpixel_refinement: true,
        }
    }

    /// Set FAST threshold
    pub fn threshold(mut self, threshold: u8) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set patch size for orientation computation
    pub fn patch_size(mut self, patch_size: usize) -> Self {
        self.config.patch_size = patch_size;
        self
    }

    /// Set number of threads for parallel processing
    pub fn threads(mut self, n_threads: usize) -> Self {
        self.config.n_threads = n_threads;
        self
    }

    /// Enable/disable SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }

    /// Enable/disable optimized memory layout
    pub fn optimized_layout(mut self, enable: bool) -> Self {
        self.enable_optimized_layout = enable;
        self
    }

    /// Enable/disable Harris corner response
    pub fn harris_corners(mut self, enable: bool) -> Self {
        self.enable_harris_corners = enable;
        self
    }

    /// Enable/disable adaptive thresholding
    pub fn adaptive_thresholding(mut self, enable: bool) -> Self {
        self.enable_adaptive_thresholding = enable;
        self
    }

    /// Enable/disable CLAHE preprocessing
    pub fn clahe_preprocessing(mut self, enable: bool) -> Self {
        self.enable_clahe_preprocessing = enable;
        self
    }

    /// Set minimum distance for non-maximum suppression
    pub fn nms_distance(mut self, distance: f32) -> Self {
        self.nms_distance = distance;
        self
    }

    /// Enable/disable subpixel refinement
    pub fn subpixel_refinement(mut self, enable: bool) -> Self {
        self.subpixel_refinement = enable;
        self
    }

    /// Apply fast preset (optimized for speed)
    pub fn preset_fast(mut self) -> Self {
        self.config.threshold = 30;
        self.config.patch_size = 15;
        self.config.n_threads = num_cpus::get();
        self.enable_simd = true;
        self.enable_optimized_layout = true;
        self.enable_harris_corners = false;
        self.enable_adaptive_thresholding = false;
        self.enable_clahe_preprocessing = false;
        self.nms_distance = 5.0;
        self.subpixel_refinement = false;
        self
    }

    /// Apply quality preset (balanced quality and performance)
    pub fn preset_quality(mut self) -> Self {
        self.config.threshold = 15;
        self.config.patch_size = 31;
        self.config.n_threads = num_cpus::get();
        self.enable_simd = true;
        self.enable_optimized_layout = true;
        self.enable_harris_corners = true;
        self.enable_adaptive_thresholding = true;
        self.enable_clahe_preprocessing = false;
        self.nms_distance = 2.0;
        self.subpixel_refinement = true;
        self
    }

    /// Apply illumination robust preset (maximum robustness)
    pub fn preset_illumination_robust(mut self) -> Self {
        self.config.threshold = 12;
        self.config.patch_size = 31;
        self.config.n_threads = num_cpus::get();
        self.enable_simd = true;
        self.enable_optimized_layout = true;
        self.enable_harris_corners = true;
        self.enable_adaptive_thresholding = true;
        self.enable_clahe_preprocessing = true;
        self.nms_distance = 2.0;
        self.subpixel_refinement = true;
        self
    }

    /// Build configured detector
    pub fn build(self) -> FastResult<ConfiguredDetector> {
        let detector = FastDetector::new(self.config.clone(), self.width, self.height)?;
        Ok(ConfiguredDetector {
            detector,
            config: self,
        })
    }

    /// Generate summary of current configuration
    pub fn summary(&self) -> String {
        format!(
            "DetectorBuilder: {}x{}, threshold={}, patch_size={}, threads={}, features=[SIMD:{}, OptLayout:{}, Harris:{}, Adaptive:{}, CLAHE:{}], NMS:{:.1}, Subpixel:{}",
            self.width, self.height, self.config.threshold, self.config.patch_size, self.config.n_threads,
            self.enable_simd, self.enable_optimized_layout, self.enable_harris_corners,
            self.enable_adaptive_thresholding, self.enable_clahe_preprocessing,
            self.nms_distance, self.subpixel_refinement
        )
    }

    /// Create builder from existing configuration
    pub fn from_config(config: DetectorConfig) -> Self {
        Self {
            config: config.core,
            width: config.width,
            height: config.height,
            enable_simd: config.enable_simd,
            enable_optimized_layout: config.enable_optimized_layout,
            enable_harris_corners: config.enable_harris_corners,
            enable_adaptive_thresholding: config.enable_adaptive_thresholding,
            enable_clahe_preprocessing: config.enable_clahe_preprocessing,
            nms_distance: config.nms_distance,
            subpixel_refinement: config.subpixel_refinement,
        }
    }

    /// Convert to DetectorConfig
    pub fn to_config(self) -> DetectorConfig {
        DetectorConfig {
            core: self.config,
            width: self.width,
            height: self.height,
            enable_simd: self.enable_simd,
            enable_optimized_layout: self.enable_optimized_layout,
            enable_harris_corners: self.enable_harris_corners,
            enable_adaptive_thresholding: self.enable_adaptive_thresholding,
            enable_clahe_preprocessing: self.enable_clahe_preprocessing,
            nms_distance: self.nms_distance,
            subpixel_refinement: self.subpixel_refinement,
            name: None,
            description: None,
            version: None,
        }
    }
}

/// Pre-configured detector with feature-aware detection
pub struct ConfiguredDetector {
    detector: FastDetector,
    config: DetectorBuilder,
}

impl ConfiguredDetector {
    /// Detect keypoints with full feature pipeline
    pub fn detect_keypoints(&self, img: &Image) -> FastResult<Vec<Keypoint>> {
        // Apply CLAHE preprocessing if enabled
        let processed_img = if self.config.enable_clahe_preprocessing {
            crate::preprocessing::ImagePreprocessing::apply_clahe_preprocessing(img, self.config.width, self.config.height)?
        } else {
            img.clone()
        };

        // Detect keypoints with response scores
        let scored_keypoints = self.detector.detect_keypoints_with_response(&processed_img)?;
        
        // Apply non-maximum suppression
        let suppressed = crate::refinement::KeypointRefinement::non_maximum_suppression(&scored_keypoints, self.config.nms_distance);
        
        // Apply subpixel refinement if enabled
        let refined_keypoints = if self.config.subpixel_refinement {
            suppressed
                .into_iter()
                .filter_map(|sk| crate::refinement::KeypointRefinement::refine_keypoint_subpixel(&processed_img, self.config.width, self.config.height, sk.keypoint).ok())
                .collect()
        } else {
            suppressed.into_iter().map(|sk| sk.keypoint).collect()
        };
        
        Ok(refined_keypoints)
    }

    /// Detect keypoints with response scores
    pub fn detect_keypoints_with_response(&self, img: &Image) -> FastResult<Vec<ScoredKeypoint>> {
        let processed_img = if self.config.enable_clahe_preprocessing {
            crate::preprocessing::ImagePreprocessing::apply_clahe_preprocessing(img, self.config.width, self.config.height)?
        } else {
            img.clone()
        };

        self.detector.detect_keypoints_with_response(&processed_img)
    }

    /// Get reference to underlying detector
    pub fn detector(&self) -> &FastDetector {
        &self.detector
    }

    /// Get configuration summary
    pub fn config_summary(&self) -> String {
        self.config.summary()
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.config.width, self.config.height)
    }

    /// Get scale levels for pyramid detection
    pub fn scale_levels(&self) -> &[ScaleLevel] {
        self.detector.get_scale_levels()
    }
} 