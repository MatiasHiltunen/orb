use orb_core::OrbConfig;
use crate::error::FastResult;
use crate::detector::FastDetector;
use crate::config::DetectorConfig;
use crate::configured_detector::ConfiguredDetector;

/// Builder for creating a `ConfiguredDetector`
#[derive(Debug, Clone)]
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
    /// Create a new builder with default settings
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            config: OrbConfig::default(),
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

    /// Set the FAST threshold (0-127)
    pub fn threshold(mut self, threshold: u8) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set the patch size for orientation calculation
    pub fn patch_size(mut self, patch_size: usize) -> Self {
        self.config.patch_size = patch_size;
        self
    }

    /// Set the number of threads for parallel processing
    pub fn threads(mut self, n_threads: usize) -> Self {
        self.config.n_threads = n_threads;
        self
    }

    /// Enable or disable SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }

    /// Enable or disable optimized memory layout
    pub fn optimized_layout(mut self, enable: bool) -> Self {
        self.enable_optimized_layout = enable;
        self
    }

    /// Enable or disable Harris corner scoring
    pub fn harris_corners(mut self, enable: bool) -> Self {
        self.enable_harris_corners = enable;
        self
    }

    /// Enable or disable adaptive thresholding
    pub fn adaptive_thresholding(mut self, enable: bool) -> Self {
        self.enable_adaptive_thresholding = enable;
        self
    }

    /// Enable or disable CLAHE preprocessing
    pub fn clahe_preprocessing(mut self, enable: bool) -> Self {
        self.enable_clahe_preprocessing = enable;
        self
    }

    /// Set the non-maximum suppression (NMS) distance
    pub fn nms_distance(mut self, distance: f32) -> Self {
        self.nms_distance = distance;
        self
    }

    /// Enable or disable subpixel refinement
    pub fn subpixel_refinement(mut self, enable: bool) -> Self {
        self.subpixel_refinement = enable;
        self
    }

    /// Set the FAST variant (e.g., FAST-9, FAST-12)
    pub fn fast_variant(mut self, variant: orb_core::FastVariant) -> Self {
        self.config.fast_variant = variant;
        self
    }

    /// Set a custom FAST-N variant
    pub fn fast_n(mut self, n: u8) -> Self {
        self.config.fast_variant = orb_core::FastVariant::custom(n);
        self
    }
    
    /// Set the NMS radius (alternative to nms_distance)
    pub fn nms_radius(mut self, radius: f32) -> Self {
        self.config.nms_radius = radius;
        self
    }

    /// Apply the ultra-fast preset
    pub fn preset_ultra_fast(mut self) -> Self {
        let preset = DetectorConfig::ultra_fast_preset(self.width, self.height);
        self.config = preset.core;
        self.nms_distance = preset.nms_distance;
        self.subpixel_refinement = preset.subpixel_refinement;
        self
    }
    
    /// Apply the precision preset
    pub fn preset_precision(mut self) -> Self {
        let preset = DetectorConfig::precision_preset(self.width, self.height);
        self.config = preset.core;
        self.nms_distance = preset.nms_distance;
        self.subpixel_refinement = preset.subpixel_refinement;
        self
    }

    /// Apply the balanced preset
    pub fn preset_balanced(mut self) -> Self {
        let preset = DetectorConfig::balanced_preset(self.width, self.height);
        self.config = preset.core;
        self.nms_distance = preset.nms_distance;
        self.subpixel_refinement = preset.subpixel_refinement;
        self
    }
    
    /// Build the `ConfiguredDetector`
    pub fn build(self) -> FastResult<ConfiguredDetector> {
        let config = self.clone().to_config();
        let detector = FastDetector::new(config.core.clone(), config.width, config.height)?;
        Ok(ConfiguredDetector {
            detector,
            config: self,
        })
    }

    /// Generate a summary of the builder's configuration
    pub fn summary(&self) -> String {
        self.clone().to_config().summary()
    }
    
    /// Create a builder from an existing `DetectorConfig`
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
    
    /// Convert the builder into a `DetectorConfig`
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

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
} 