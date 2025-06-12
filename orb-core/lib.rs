/// Row-major 8-bit grayscale image
pub type Image = Vec<u8>;

/// Key-point ≙ FAST corner + orientation (radians) with subpixel precision
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,      // Subpixel x coordinate
    pub y: f32,      // Subpixel y coordinate
    pub angle: f32,
}

/// 256-bit binary descriptor = 32 bytes
pub type Descriptor = [u8; 32];

#[derive(Debug, Clone)]
pub struct OrbConfig {
    pub threshold: u8,
    pub patch_size: usize,
    pub n_threads: usize,
}

impl Default for OrbConfig {
    fn default() -> Self {
        Self {
            threshold: 20,
            patch_size: 15,
            n_threads: num_cpus::get().max(1),
        }
    }
}

/// Initialize Rayon thread pool with the specified number of threads
pub fn init_thread_pool(n_threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
} 