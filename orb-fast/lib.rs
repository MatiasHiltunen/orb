use orb_core::{Image, Keypoint, OrbConfig};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub enum FastError {
    InvalidImageSize { width: usize, height: usize },
    InvalidImageData { expected_len: usize, actual_len: usize },
    InvalidThreshold(u8),
    InvalidPatchSize { patch_size: usize, min_image_dim: usize },
    ImageTooSmall { width: usize, height: usize, min_size: usize },
}

impl std::fmt::Display for FastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FastError::InvalidImageSize { width, height } => {
                write!(f, "Invalid image dimensions: {}x{} (must be > 0)", width, height)
            }
            FastError::InvalidImageData { expected_len, actual_len } => {
                write!(f, "Image data length mismatch: expected {}, got {}", expected_len, actual_len)
            }
            FastError::InvalidThreshold(t) => {
                write!(f, "Invalid threshold: {} (must be 1-127)", t)
            }
            FastError::InvalidPatchSize { patch_size, min_image_dim } => {
                write!(f, "Patch size {} too large for minimum image dimension {}", patch_size, min_image_dim)
            }
            FastError::ImageTooSmall { width, height, min_size } => {
                write!(f, "Image {}x{} too small (minimum {}x{})", width, height, min_size, min_size)
            }
        }
    }
}

impl std::error::Error for FastError {}

pub type FastResult<T> = Result<T, FastError>;

/// Keypoint with corner response score for NMS
#[derive(Debug, Clone, Copy)]
pub struct ScoredKeypoint {
    pub keypoint: Keypoint,
    pub response: f32,
}

pub struct FastDetector {
    cfg: OrbConfig,
    w: usize,
    h: usize,
}

impl FastDetector {
    /// Creates a new FAST detector with validation
    pub fn new(cfg: OrbConfig, width: usize, height: usize) -> FastResult<Self> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(FastError::InvalidImageSize { width, height });
        }

        // FAST requires at least 7x7 image (3-pixel border on each side)
        const MIN_SIZE: usize = 7;
        if width < MIN_SIZE || height < MIN_SIZE {
            return Err(FastError::ImageTooSmall { 
                width, height, min_size: MIN_SIZE 
            });
        }

        // Validate threshold (0 would detect everything, >127 could cause issues with u8 arithmetic)
        if cfg.threshold == 0 || cfg.threshold > 127 {
            return Err(FastError::InvalidThreshold(cfg.threshold));
        }

        // Validate patch size
        if cfg.patch_size % 2 == 0 {
            return Err(FastError::InvalidPatchSize { 
                patch_size: cfg.patch_size, 
                min_image_dim: std::cmp::min(width, height) 
            });
        }

        let min_dim = std::cmp::min(width, height);
        if cfg.patch_size >= min_dim {
            return Err(FastError::InvalidPatchSize { 
                patch_size: cfg.patch_size, 
                min_image_dim: min_dim 
            });
        }

        Ok(Self {
            cfg,
            w: width,
            h: height,
        })
    }

    /// Validates image data before processing
    fn validate_image(&self, img: &Image) -> FastResult<()> {
        let expected_len = self.w * self.h;
        if img.len() != expected_len {
            return Err(FastError::InvalidImageData {
                expected_len,
                actual_len: img.len(),
            });
        }
        Ok(())
    }

    pub fn detect_keypoints(&self, img: &Image) -> FastResult<Vec<Keypoint>> {
        let scored_keypoints = self.detect_keypoints_with_response(img)?;
        let suppressed = self.non_maximum_suppression(&scored_keypoints, 3.0);
        Ok(suppressed.into_iter().map(|sk| sk.keypoint).collect())
    }

    /// Detect keypoints with corner response scores
    pub fn detect_keypoints_with_response(&self, img: &Image) -> FastResult<Vec<ScoredKeypoint>> {
        // Validate input
        self.validate_image(img)?;

        const OFF: [(i32, i32); 16] = [
            (-3, 0), (-3, 1), (-2, 2), (-1, 3),
            (0, 3), (1, 3), (2, 2), (3, 1),
            (3, 0), (3, -1), (2, -2), (1, -3),
            (0, -3), (-1, -3), (-2, -2), (-3, -1),
        ];

        let rows = 3..self.h.saturating_sub(3);
        let keypoints = rows.into_par_iter()
            .flat_map_iter(|y| {
                let mut v = Vec::new();
                for x in 3..self.w - 3 {
                    let p = img[y * self.w + x];
                    let mut bri = 0;
                    let mut drk = 0;
                    let mut bri_sum = 0i32;
                    let mut drk_sum = 0i32;
                    
                    for &(dx, dy) in &OFF {
                        let xx = (x as i32 + dx).clamp(0, (self.w - 1) as i32) as usize;
                        let yy = (y as i32 + dy).clamp(0, (self.h - 1) as i32) as usize;
                        let q = img[yy * self.w + xx];
                        
                        if q >= p.saturating_add(self.cfg.threshold) {
                            bri += 1;
                            bri_sum += (q as i32) - (p as i32);
                        } else if q.saturating_add(self.cfg.threshold) <= p {
                            drk += 1;
                            drk_sum += (p as i32) - (q as i32);
                        }
                    }
                    
                    if bri >= 12 || drk >= 12 {
                        match self.compute_orientation(img, x, y) {
                            Ok(angle) => {
                                // Compute corner response as maximum intensity difference
                                let response = if bri >= 12 {
                                    bri_sum as f32 / bri as f32
                                } else {
                                    drk_sum as f32 / drk as f32
                                };
                                
                                v.push(ScoredKeypoint {
                                    keypoint: Keypoint { x, y, angle },
                                    response: response.abs(),
                                });
                            }
                            Err(_) => {} // Skip this keypoint if orientation computation fails
                        }
                    }
                }
                v
            })
            .collect();

        Ok(keypoints)
    }

    /// Non-Maximum Suppression to remove nearby redundant keypoints
    fn non_maximum_suppression(&self, keypoints: &[ScoredKeypoint], min_distance: f32) -> Vec<ScoredKeypoint> {
        if keypoints.is_empty() {
            return Vec::new();
        }

        // Sort keypoints by response (highest first)
        let mut sorted_keypoints = keypoints.to_vec();
        sorted_keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));

        let mut suppressed: Vec<ScoredKeypoint> = Vec::new();
        let min_distance_sq = min_distance * min_distance;

        for candidate in sorted_keypoints {
            let mut is_local_maximum = true;
            
            // Check if this keypoint is too close to any already accepted keypoint
            for accepted in &suppressed {
                let dx = candidate.keypoint.x as f32 - accepted.keypoint.x as f32;
                let dy = candidate.keypoint.y as f32 - accepted.keypoint.y as f32;
                let distance_sq = dx * dx + dy * dy;
                
                if distance_sq < min_distance_sq {
                    is_local_maximum = false;
                    break;
                }
            }
            
            if is_local_maximum {
                suppressed.push(candidate);
            }
        }

        suppressed
    }

    /// Compute orientation with improved safety checks
    fn compute_orientation(&self, img: &Image, x: usize, y: usize) -> FastResult<f32> {
        let half = (self.cfg.patch_size / 2) as i32;
        let (cx, cy) = (x as i32, y as i32);
        
        // Check if patch fits within image bounds
        if cx - half < 0 || cy - half < 0 || 
           cx + half >= self.w as i32 || cy + half >= self.h as i32 {
            return Ok(0.0); // Default angle for edge cases
        }

        let mut m10 = 0i64; // Use i64 to prevent overflow
        let mut m01 = 0i64;

        // Use safer, non-SIMD implementation for now to ensure correctness
        for dy in -half..=half {
            let yy = (cy + dy) as usize;
            for dx in -half..=half {
                let xx = (cx + dx) as usize;
                let val = img[yy * self.w + xx] as i64;
                m10 += dx as i64 * val;
                m01 += dy as i64 * val;
            }
        }

        Ok((m01 as f32).atan2(m10 as f32))
    }

    /// Get detector configuration
    pub fn config(&self) -> &OrbConfig {
        &self.cfg
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.w, self.h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> OrbConfig {
        OrbConfig {
            threshold: 20,
            patch_size: 15,
            n_threads: 1,
        }
    }

    fn create_small_test_config() -> OrbConfig {
        OrbConfig {
            threshold: 20,
            patch_size: 5, // Smaller patch for small images
            n_threads: 1,
        }
    }

    fn create_test_image(width: usize, height: usize) -> Image {
        vec![128; width * height] // Gray image
    }

    fn create_corner_image(width: usize, height: usize) -> Image {
        let mut img = vec![50; width * height]; // Dark background
        // Create a bright corner pattern in the center with stronger contrast
        let cx = width / 2;
        let cy = height / 2;
        
        // Create a square pattern that should trigger FAST detection
        for dy in -4..=4 {
            for dx in -4..=4 {
                let x = (cx as i32 + dx) as usize;
                let y = (cy as i32 + dy) as usize;
                if x < width && y < height {
                    // Create a bright square in the center
                    if dx.abs() <= 2 && dy.abs() <= 2 {
                        img[y * width + x] = 255; // Very bright
                    }
                }
            }
        }
        img
    }

    fn create_multiple_corners_image(width: usize, height: usize) -> Image {
        let mut img = vec![50; width * height];
        
        // Create multiple corner patterns
        let corners = vec![(width/4, height/4), (3*width/4, height/4), (width/2, height/2)];
        
        for &(cx, cy) in &corners {
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let x = (cx as i32 + dx) as usize;
                    let y = (cy as i32 + dy) as usize;
                    if x < width && y < height {
                        if dx.abs() <= 1 && dy.abs() <= 1 {
                            img[y * width + x] = 255;
                        }
                    }
                }
            }
        }
        img
    }

    #[test]
    fn test_valid_constructor() {
        let detector = FastDetector::new(create_test_config(), 100, 100);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_invalid_dimensions() {
        // Zero width
        let result = FastDetector::new(create_test_config(), 0, 100);
        assert!(matches!(result, Err(FastError::InvalidImageSize { .. })));

        // Zero height
        let result = FastDetector::new(create_test_config(), 100, 0);
        assert!(matches!(result, Err(FastError::InvalidImageSize { .. })));
    }

    #[test]
    fn test_too_small_image() {
        let result = FastDetector::new(create_test_config(), 6, 6);
        assert!(matches!(result, Err(FastError::ImageTooSmall { .. })));
    }

    #[test]
    fn test_invalid_threshold() {
        let mut cfg = create_test_config();
        
        // Zero threshold
        cfg.threshold = 0;
        let result = FastDetector::new(cfg.clone(), 100, 100);
        assert!(matches!(result, Err(FastError::InvalidThreshold(0))));

        // Too high threshold
        cfg.threshold = 200;
        let result = FastDetector::new(cfg, 100, 100);
        assert!(matches!(result, Err(FastError::InvalidThreshold(200))));
    }

    #[test]
    fn test_invalid_patch_size() {
        let mut cfg = create_test_config();
        
        // Even patch size
        cfg.patch_size = 16;
        let result = FastDetector::new(cfg.clone(), 100, 100);
        assert!(matches!(result, Err(FastError::InvalidPatchSize { .. })));

        // Patch size too large for image
        cfg.patch_size = 101;
        let result = FastDetector::new(cfg, 100, 100);
        assert!(matches!(result, Err(FastError::InvalidPatchSize { .. })));
    }

    #[test]
    fn test_invalid_image_data() {
        let detector = FastDetector::new(create_small_test_config(), 10, 10).unwrap();
        
        // Wrong size image
        let img = vec![0; 50]; // Should be 100
        let result = detector.detect_keypoints(&img);
        assert!(matches!(result, Err(FastError::InvalidImageData { .. })));
    }

    #[test]
    fn test_empty_image_detection() {
        let detector = FastDetector::new(create_small_test_config(), 10, 10).unwrap();
        let img = create_test_image(10, 10);
        
        let result = detector.detect_keypoints(&img);
        assert!(result.is_ok());
        let keypoints = result.unwrap();
        // Uniform image should have no corners
        assert_eq!(keypoints.len(), 0);
    }

    #[test]
    fn test_corner_detection() {
        let detector = FastDetector::new(create_small_test_config(), 20, 20).unwrap();
        let img = create_corner_image(20, 20);
        
        let result = detector.detect_keypoints(&img);
        assert!(result.is_ok());
        let keypoints = result.unwrap();
        // Should detect the bright corner we created
        assert!(keypoints.len() > 0);
    }

    #[test]
    fn test_non_maximum_suppression() {
        let detector = FastDetector::new(create_test_config(), 50, 50).unwrap();
        let img = create_multiple_corners_image(50, 50);
        
        // Test with response scores
        let scored_keypoints = detector.detect_keypoints_with_response(&img).unwrap();
        let suppressed = detector.non_maximum_suppression(&scored_keypoints, 5.0);
        
        // NMS should reduce the number of keypoints
        assert!(suppressed.len() <= scored_keypoints.len());
        
        // Verify minimum distance between remaining keypoints
        for i in 0..suppressed.len() {
            for j in (i+1)..suppressed.len() {
                let dx = suppressed[i].keypoint.x as f32 - suppressed[j].keypoint.x as f32;
                let dy = suppressed[i].keypoint.y as f32 - suppressed[j].keypoint.y as f32;
                let distance = (dx * dx + dy * dy).sqrt();
                assert!(distance >= 5.0, "Keypoints too close after NMS: {}", distance);
            }
        }
    }

    #[test]
    fn test_configuration_access() {
        let cfg = create_test_config();
        let detector = FastDetector::new(cfg.clone(), 20, 20).unwrap();
        
        assert_eq!(detector.config().threshold, cfg.threshold);
        assert_eq!(detector.config().patch_size, cfg.patch_size);
        assert_eq!(detector.dimensions(), (20, 20));
    }

    #[test]
    fn test_boundary_cases() {
        // Minimum valid size
        let detector = FastDetector::new(create_small_test_config(), 7, 7).unwrap();
        let img = create_test_image(7, 7);
        let result = detector.detect_keypoints(&img);
        assert!(result.is_ok());
    }

    #[test]
    fn test_orientation_computation() {
        let detector = FastDetector::new(create_small_test_config(), 20, 20).unwrap();
        let img = create_corner_image(20, 20);
        
        // Test orientation computation directly
        let result = detector.compute_orientation(&img, 10, 10);
        assert!(result.is_ok());
        let angle = result.unwrap();
        // Angle should be valid
        assert!(angle.is_finite());
    }

    #[test]
    fn test_parallel_safety() {
        let detector = FastDetector::new(create_test_config(), 100, 100).unwrap();
        let img = create_corner_image(100, 100);
        
        // Run detection multiple times to test parallel safety
        for _ in 0..10 {
            let result = detector.detect_keypoints(&img);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_response_scoring() {
        let detector = FastDetector::new(create_small_test_config(), 20, 20).unwrap();
        let img = create_corner_image(20, 20);
        
        let scored_keypoints = detector.detect_keypoints_with_response(&img).unwrap();
        
        // All keypoints should have positive responses
        for sk in &scored_keypoints {
            assert!(sk.response > 0.0);
            assert!(sk.response.is_finite());
        }
    }
} 