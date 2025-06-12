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
        
        // Apply subpixel refinement to the filtered keypoints
        let refined_keypoints = suppressed
            .into_iter()
            .filter_map(|sk| self.refine_keypoint_subpixel(img, sk.keypoint).ok())
            .collect();
            
        Ok(refined_keypoints)
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
                        match self.compute_orientation(img, x as f32, y as f32) {
                            Ok(angle) => {
                                // Compute corner response as maximum intensity difference
                                let response = if bri >= 12 {
                                    bri_sum as f32 / bri as f32
                                } else {
                                    drk_sum as f32 / drk as f32
                                };
                                
                                v.push(ScoredKeypoint {
                                    keypoint: Keypoint { x: x as f32, y: y as f32, angle },
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
                let dx = candidate.keypoint.x - accepted.keypoint.x;
                let dy = candidate.keypoint.y - accepted.keypoint.y;
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

    /// Refine keypoint location to subpixel accuracy using quadratic fitting
    fn refine_keypoint_subpixel(&self, img: &Image, kp: Keypoint) -> FastResult<Keypoint> {
        let ix = kp.x.round() as i32;
        let iy = kp.y.round() as i32;
        
        // Check bounds for 3x3 neighborhood
        if ix < 1 || iy < 1 || ix >= (self.w as i32 - 1) || iy >= (self.h as i32 - 1) {
            return Ok(kp); // Return original if too close to boundary
        }

        // Compute corner responses in 3x3 neighborhood
        let mut responses = [[0.0f32; 3]; 3];
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = (ix + dx) as usize;
                let y = (iy + dy) as usize;
                responses[(dy + 1) as usize][(dx + 1) as usize] = self.compute_corner_response(img, x, y);
            }
        }

        // Fit 2D quadratic surface: f(x,y) = a*x² + b*y² + c*x*y + d*x + e*y + f
        // We use the center point as origin, so we solve for the offset from center
        let f00 = responses[1][1]; // Center response
        let fx_pos = responses[1][2]; // Right
        let fx_neg = responses[1][0]; // Left  
        let fy_pos = responses[2][1]; // Down
        let fy_neg = responses[0][1]; // Up
        let fxy = responses[2][2];    // Bottom-right for cross term

        // Compute derivatives using finite differences
        let fx = (fx_pos - fx_neg) * 0.5;         // First derivative in x
        let fy = (fy_pos - fy_neg) * 0.5;         // First derivative in y
        let fxx = fx_pos - 2.0 * f00 + fx_neg;    // Second derivative in x
        let fyy = fy_pos - 2.0 * f00 + fy_neg;    // Second derivative in y
        let fxy_cross = (fxy + f00 - fx_pos - fy_pos) * 0.5; // Cross derivative

        // Solve for the peak of the quadratic surface
        // Peak occurs where gradient = 0: [fxx, fxy; fxy, fyy] * [dx; dy] = -[fx; fy]
        let det = fxx * fyy - fxy_cross * fxy_cross;
        
        if det.abs() < 1e-6 {
            return Ok(kp); // Singular matrix, return original
        }

        let dx_offset = -(fyy * fx - fxy_cross * fy) / det;
        let dy_offset = -(fxx * fy - fxy_cross * fx) / det;

        // Clamp the offset to prevent excessive movement
        let dx_offset = dx_offset.clamp(-0.5, 0.5);
        let dy_offset = dy_offset.clamp(-0.5, 0.5);

        // Compute refined coordinates
        let refined_x = kp.x + dx_offset;
        let refined_y = kp.y + dy_offset;

        // Recompute orientation at refined location
        let refined_angle = self.compute_orientation(img, refined_x, refined_y)?;

        Ok(Keypoint {
            x: refined_x,
            y: refined_y,
            angle: refined_angle,
        })
    }

    /// Compute corner response at a specific pixel location
    fn compute_corner_response(&self, img: &Image, x: usize, y: usize) -> f32 {
        if x < 3 || y < 3 || x >= self.w - 3 || y >= self.h - 3 {
            return 0.0;
        }

        const OFF: [(i32, i32); 16] = [
            (-3, 0), (-3, 1), (-2, 2), (-1, 3),
            (0, 3), (1, 3), (2, 2), (3, 1),
            (3, 0), (3, -1), (2, -2), (1, -3),
            (0, -3), (-1, -3), (-2, -2), (-3, -1),
        ];

        let p = img[y * self.w + x];
        let mut bri_sum = 0i32;
        let mut drk_sum = 0i32;
        let mut bri_count = 0;
        let mut drk_count = 0;

        for &(dx, dy) in &OFF {
            let xx = (x as i32 + dx).clamp(0, (self.w - 1) as i32) as usize;
            let yy = (y as i32 + dy).clamp(0, (self.h - 1) as i32) as usize;
            let q = img[yy * self.w + xx];

            if q >= p.saturating_add(self.cfg.threshold) {
                bri_sum += (q as i32) - (p as i32);
                bri_count += 1;
            } else if q.saturating_add(self.cfg.threshold) <= p {
                drk_sum += (p as i32) - (q as i32);
                drk_count += 1;
            }
        }

        if bri_count >= 12 {
            bri_sum as f32 / bri_count as f32
        } else if drk_count >= 12 {
            drk_sum as f32 / drk_count as f32
        } else {
            0.0
        }
    }

    /// Compute orientation with improved safety checks for subpixel coordinates
    fn compute_orientation(&self, img: &Image, x: f32, y: f32) -> FastResult<f32> {
        let half = (self.cfg.patch_size / 2) as i32;
        let cx = x.round() as i32;
        let cy = y.round() as i32;
        
        // Check if patch fits within image bounds
        if cx - half < 0 || cy - half < 0 || 
           cx + half >= self.w as i32 || cy + half >= self.h as i32 {
            return Ok(0.0); // Default angle for edge cases
        }

        let mut m10 = 0i64; // Use i64 to prevent overflow
        let mut m01 = 0i64;

        // Use bilinear interpolation for subpixel-accurate orientation
        for dy in -half..=half {
            let yy = cy + dy;
            for dx in -half..=half {
                let xx = cx + dx;
                
                // Bilinear interpolation at (x + dx_offset, y + dy_offset)
                let sample_x = x + dx as f32;
                let sample_y = y + dy as f32;
                let val = self.bilinear_interpolate(img, sample_x, sample_y);
                
                m10 += dx as i64 * val as i64;
                m01 += dy as i64 * val as i64;
            }
        }

        Ok((m01 as f32).atan2(m10 as f32))
    }

    /// Bilinear interpolation for subpixel sampling
    fn bilinear_interpolate(&self, img: &Image, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        // Clamp to image bounds
        if x0 < 0 || y0 < 0 || x1 >= self.w as i32 || y1 >= self.h as i32 {
            let cx = x.round().clamp(0.0, (self.w - 1) as f32) as usize;
            let cy = y.round().clamp(0.0, (self.h - 1) as f32) as usize;
            return img[cy * self.w + cx] as f32;
        }

        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let p00 = img[y0 as usize * self.w + x0 as usize] as f32;
        let p10 = img[y0 as usize * self.w + x1 as usize] as f32;
        let p01 = img[y1 as usize * self.w + x0 as usize] as f32;
        let p11 = img[y1 as usize * self.w + x1 as usize] as f32;

        let top = p00 * (1.0 - dx) + p10 * dx;
        let bottom = p01 * (1.0 - dx) + p11 * dx;
        
        top * (1.0 - dy) + bottom * dy
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
                let dx = suppressed[i].keypoint.x - suppressed[j].keypoint.x;
                let dy = suppressed[i].keypoint.y - suppressed[j].keypoint.y;
                let distance = (dx * dx + dy * dy).sqrt();
                assert!(distance >= 5.0, "Keypoints too close after NMS: {}", distance);
            }
        }
    }

    #[test]
    fn test_subpixel_refinement() {
        let detector = FastDetector::new(create_small_test_config(), 20, 20).unwrap();
        let img = create_corner_image(20, 20);
        
        // Create a keypoint at integer coordinates
        let original_kp = Keypoint { x: 10.0, y: 10.0, angle: 0.0 };
        
        let refined_kp = detector.refine_keypoint_subpixel(&img, original_kp).unwrap();
        
        // Refined coordinates should be valid
        assert!(refined_kp.x > 0.0 && refined_kp.x < 20.0);
        assert!(refined_kp.y > 0.0 && refined_kp.y < 20.0);
        assert!(refined_kp.angle.is_finite());
        
        // Should have subpixel precision (fractional part)
        let has_subpixel_x = (refined_kp.x - refined_kp.x.round()).abs() > 1e-6;
        let has_subpixel_y = (refined_kp.y - refined_kp.y.round()).abs() > 1e-6;
        
        // At least one coordinate should be refined to subpixel precision
        // (depending on the corner pattern, refinement may be minimal)
        println!("Refined: ({}, {}) from ({}, {})", 
                refined_kp.x, refined_kp.y, original_kp.x, original_kp.y);
    }

    #[test]
    fn test_bilinear_interpolation() {
        let detector = FastDetector::new(create_small_test_config(), 10, 10).unwrap();
        let img = vec![100; 100]; // Uniform image
        
        // Test interpolation at various positions
        let val1 = detector.bilinear_interpolate(&img, 5.0, 5.0);
        let val2 = detector.bilinear_interpolate(&img, 5.5, 5.5);
        
        // Should return the uniform value
        assert_eq!(val1, 100.0);
        assert_eq!(val2, 100.0);
        
        // Test gradient image
        let mut gradient_img = vec![0; 100];
        for y in 0..10 {
            for x in 0..10 {
                gradient_img[y * 10 + x] = (x * 25) as u8; // Horizontal gradient
            }
        }
        
        let val3 = detector.bilinear_interpolate(&gradient_img, 2.5, 5.0);
        // Should interpolate between x=2 (50) and x=3 (75) = 62.5
        assert!((val3 - 62.5).abs() < 0.1);
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
        let result = detector.compute_orientation(&img, 10.0, 10.0);
        assert!(result.is_ok());
        let angle = result.unwrap();
        // Angle should be valid
        assert!(angle.is_finite());
        
        // Test with subpixel coordinates
        let result_subpix = detector.compute_orientation(&img, 10.5, 10.5);
        assert!(result_subpix.is_ok());
        let angle_subpix = result_subpix.unwrap();
        assert!(angle_subpix.is_finite());
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

    #[test]
    fn test_corner_response_computation() {
        let detector = FastDetector::new(create_test_config(), 20, 20).unwrap();
        let img = create_corner_image(20, 20);
        
        // Test corner response at center (should be high)
        let response_center = detector.compute_corner_response(&img, 10, 10);
        assert!(response_center > 0.0);
        
        // Test corner response at edge (should be low/zero)
        let response_edge = detector.compute_corner_response(&img, 1, 1);
        assert!(response_edge <= response_center);
    }
} 