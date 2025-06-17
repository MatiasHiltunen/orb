use orb_core::OrbConfig;
use crate::error::{FastError, FastResult};
use crate::builder::DetectorBuilder;

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
                fast_variant: orb_core::FastVariant::Fast9,
                nms_radius: 3.0,
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
                fast_variant: orb_core::FastVariant::Fast9,
                nms_radius: 5.0,
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
                fast_variant: orb_core::FastVariant::Fast9,
                nms_radius: 2.0,
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
                fast_variant: orb_core::FastVariant::Fast12,
                nms_radius: 2.0,
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

    /// Ultra-fast preset optimized for maximum speed
    pub fn ultra_fast_preset(width: usize, height: usize) -> Self {
        Self {
            core: OrbConfig {
                threshold: 35,
                patch_size: 15,
                n_threads: num_cpus::get(),
                fast_variant: orb_core::FastVariant::custom(6), // Less selective for more speed
                nms_radius: 8.0,
            },
            width,
            height,
            enable_simd: true,
            enable_optimized_layout: true,
            enable_harris_corners: false,
            enable_adaptive_thresholding: false,
            enable_clahe_preprocessing: false,
            nms_distance: 8.0,
            subpixel_refinement: false,
            name: Some("Ultra Fast".to_string()),
            description: Some("Maximum speed with minimal processing - ideal for real-time applications".to_string()),
            version: Some("1.0".to_string()),
        }
    }

    /// Precision preset optimized for maximum corner quality
    pub fn precision_preset(width: usize, height: usize) -> Self {
        Self {
            core: OrbConfig {
                threshold: 8,
                patch_size: 31,
                n_threads: num_cpus::get(),
                fast_variant: orb_core::FastVariant::FastN(14), // Custom high selectivity
                nms_radius: 1.5,
            },
            width,
            height,
            enable_simd: true,
            enable_optimized_layout: true,
            enable_harris_corners: true,
            enable_adaptive_thresholding: true,
            enable_clahe_preprocessing: true,
            nms_distance: 1.5,
            subpixel_refinement: true,
            name: Some("Precision".to_string()),
            description: Some("Maximum quality with Harris-FAST hybrid scoring and enhanced preprocessing".to_string()),
            version: Some("1.0".to_string()),
        }
    }

    /// Balanced preset for general-purpose applications
    pub fn balanced_preset(width: usize, height: usize) -> Self {
        Self {
            core: OrbConfig {
                threshold: 15,
                patch_size: 23,
                n_threads: num_cpus::get(),
                fast_variant: orb_core::FastVariant::custom(10), // Balanced selectivity
                nms_radius: 2.5,
            },
            width,
            height,
            enable_simd: true,
            enable_optimized_layout: true,
            enable_harris_corners: true,
            enable_adaptive_thresholding: true,
            enable_clahe_preprocessing: false,
            nms_distance: 2.5,
            subpixel_refinement: true,
            name: Some("Balanced".to_string()),
            description: Some("Optimal balance of speed and quality for most applications".to_string()),
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

    /// Generate comprehensive summary with variant information
    pub fn summary(&self) -> String {
        format!(
            "DetectorConfig: {}x{}, variant={}, threshold={}, NMS={:.1}, features=[SIMD:{}, OptLayout:{}, Harris:{}, Adaptive:{}, CLAHE:{}, Subpixel:{}]",
            self.width, self.height,
            self.core.fast_variant.name(),
            self.core.threshold,
            self.nms_distance,
            self.enable_simd,
            self.enable_optimized_layout,
            self.enable_harris_corners,
            self.enable_adaptive_thresholding,
            self.enable_clahe_preprocessing,
            self.subpixel_refinement
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

    /// Save configuration to a JSON file
    #[cfg(feature = "serde")]
    pub fn save_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load configuration from a JSON file
    #[cfg(feature = "serde")]
    pub fn load_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Save configuration to a TOML file
    #[cfg(feature = "serde")]
    pub fn save_toml<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let toml = self.to_toml()?;
        std::fs::write(path, toml)?;
        Ok(())
    }

    /// Load configuration from a TOML file
    #[cfg(feature = "serde")]
    pub fn load_toml<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let toml_str = std::fs::read_to_string(path)?;
        Self::from_toml(&toml_str)
    }

    /// Serialize configuration to a JSON string
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize configuration from a JSON string
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: DetectorConfig = serde_json::from_str(json)?;
        Ok(config)
    }

    /// Serialize configuration to a TOML string
    #[cfg(feature = "serde")]
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Deserialize configuration from a TOML string
    #[cfg(feature = "serde")]
    pub fn from_toml(toml_str: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: DetectorConfig = toml::from_str(toml_str)?;
        Ok(config)
    }
} 