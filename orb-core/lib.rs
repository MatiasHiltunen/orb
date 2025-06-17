/// Row-major 8-bit grayscale image
pub type Image = Vec<u8>;

/// Key-point ≙ FAST corner + orientation (radians) with subpixel precision
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,      // Subpixel x coordinate
    pub y: f32,      // Subpixel y coordinate
    pub angle: f32,
}

/// 256-bit binary descriptor = 32 bytes
pub type Descriptor = [u8; 32];

/// FAST corner detection variant
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastVariant {
    /// FAST-9: requires 9 consecutive pixels
    Fast9,
    /// FAST-12: requires 12 consecutive pixels (more selective)
    Fast12,
    /// FAST-16: requires 16 consecutive pixels (most selective)
    Fast16,
    /// Custom FAST-N: requires N consecutive pixels (3 ≤ N ≤ 16)
    FastN(u8),
}

impl Default for FastVariant {
    fn default() -> Self {
        FastVariant::Fast9
    }
}

impl FastVariant {
    /// Get the minimum consecutive pixel count for this variant
    pub fn min_consecutive_pixels(self) -> usize {
        match self {
            FastVariant::Fast9 => 9,
            FastVariant::Fast12 => 12,
            FastVariant::Fast16 => 16,
            FastVariant::FastN(n) => (n as usize).clamp(3, 16),
        }
    }

    /// Create a custom FAST-N variant
    pub fn custom(n: u8) -> Self {
        if n < 3 || n > 16 {
            FastVariant::Fast9 // Fallback to safe default
        } else {
            FastVariant::FastN(n)
        }
    }

    /// Get a human-readable name for this variant
    pub fn name(self) -> String {
        match self {
            FastVariant::Fast9 => "FAST-9".to_string(),
            FastVariant::Fast12 => "FAST-12".to_string(),
            FastVariant::Fast16 => "FAST-16".to_string(),
            FastVariant::FastN(n) => format!("FAST-{}", n),
        }
    }

    /// Check if this variant is more selective than another
    pub fn is_more_selective_than(self, other: FastVariant) -> bool {
        self.min_consecutive_pixels() > other.min_consecutive_pixels()
    }
}

/// Core ORB configuration with serialization support
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct OrbConfig {
    pub threshold: u8,
    pub patch_size: usize,
    pub n_threads: usize,
    pub fast_variant: FastVariant,
    pub nms_radius: f32,
}

impl Default for OrbConfig {
    fn default() -> Self {
        Self {
            threshold: 20,
            patch_size: 15,
            n_threads: num_cpus::get().max(1),
            fast_variant: FastVariant::Fast9,
            nms_radius: 3.0,
        }
    }
}

/// Initialize Rayon thread pool with the specified number of threads
pub fn init_thread_pool(n_threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
} 