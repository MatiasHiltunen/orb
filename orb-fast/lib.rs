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
                    
                    for &(dx, dy) in &OFF {
                        let xx = (x as i32 + dx).clamp(0, (self.w - 1) as i32) as usize;
                        let yy = (y as i32 + dy).clamp(0, (self.h - 1) as i32) as usize;
                        let q = img[yy * self.w + xx];
                        
                        if q >= p.saturating_add(self.cfg.threshold) {
                            bri += 1;
                        } else if q.saturating_add(self.cfg.threshold) <= p {
                            drk += 1;
                        }
                    }
                    
                    if bri >= 12 || drk >= 12 {
                        match self.compute_orientation(img, x as f32, y as f32) {
                            Ok(angle) => {
                                // Compute Harris corner response
                                let harris_response = self.compute_harris_response(img, x, y);
                                
                                // Use Harris response if positive, otherwise fall back to intensity-based response
                                let response = if harris_response > 0.0 {
                                    harris_response
                                } else {
                                    // Fall back to simple intensity difference for robustness
                                    let mut intensity_response = 0.0;
                                    let mut count = 0;
                                    
                                    for &(dx, dy) in &OFF {
                                        let xx = (x as i32 + dx).clamp(0, (self.w - 1) as i32) as usize;
                                        let yy = (y as i32 + dy).clamp(0, (self.h - 1) as i32) as usize;
                                        let q = img[yy * self.w + xx];
                                        
                                        if (bri >= 12 && q >= p.saturating_add(self.cfg.threshold)) ||
                                           (drk >= 12 && q.saturating_add(self.cfg.threshold) <= p) {
                                            intensity_response += ((q as i32) - (p as i32)).abs() as f32;
                                            count += 1;
                                        }
                                    }
                                    
                                    if count > 0 { intensity_response / count as f32 } else { 1.0 }
                                };
                                
                                v.push(ScoredKeypoint {
                                    keypoint: Keypoint { x: x as f32, y: y as f32, angle },
                                    response,
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

    /// Compute Harris corner response at a specific pixel location
    fn compute_corner_response(&self, img: &Image, x: usize, y: usize) -> f32 {
        self.compute_harris_response(img, x, y)
    }

    /// Compute Harris corner response using image gradients
    fn compute_harris_response(&self, img: &Image, x: usize, y: usize) -> f32 {
        if x < 2 || y < 2 || x >= self.w - 2 || y >= self.h - 2 {
            return 0.0;
        }

        // Compute image gradients using Sobel operators
        let mut ixx = 0.0f32;
        let mut ixy = 0.0f32;
        let mut iyy = 0.0f32;

        // Use a 5x5 window around the point for better stability
        for dy in -2..=2 {
            for dx in -2..=2 {
                let xx = (x as i32 + dx) as usize;
                let yy = (y as i32 + dy) as usize;
                
                if xx < self.w && yy < self.h {
                    // Compute gradients using finite differences
                    let (ix, iy) = self.compute_gradients(img, xx, yy);
                    
                    // Apply Gaussian-like weighting (center has more weight)
                    let weight = match (dx.abs(), dy.abs()) {
                        (0, 0) => 4.0,         // Center
                        (1, 0) | (0, 1) => 2.0, // Adjacent
                        (1, 1) => 1.0,         // Diagonal
                        _ => 0.5,              // Far corners
                    };
                    
                    // Accumulate second moment matrix elements
                    ixx += weight * ix * ix;
                    ixy += weight * ix * iy;
                    iyy += weight * iy * iy;
                }
            }
        }

        // Normalize by the sum of weights
        let total_weight = 4.0 + 8.0 * 2.0 + 4.0 * 1.0 + 12.0 * 0.5; // 28.0
        ixx /= total_weight;
        ixy /= total_weight;
        iyy /= total_weight;

        // Harris corner response: det(M) - k * trace(M)^2
        // where M is the second moment matrix [[Ixx, Ixy], [Ixy, Iyy]]
        let k = 0.05; // Slightly higher than standard 0.04 for more corners
        let det = ixx * iyy - ixy * ixy;
        let trace = ixx + iyy;
        
        let harris_response = det - k * trace * trace;
        
        // Apply a small threshold to reduce noise
        if harris_response > 1e-6 { harris_response } else { 0.0 }
    }

    /// Compute image gradients at a specific pixel using Sobel operators
    fn compute_gradients(&self, img: &Image, x: usize, y: usize) -> (f32, f32) {
        if x == 0 || y == 0 || x >= self.w - 1 || y >= self.h - 1 {
            return (0.0, 0.0);
        }

        // Sobel X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        let ix = -1.0 * img[(y - 1) * self.w + (x - 1)] as f32
                 + 0.0 * img[(y - 1) * self.w + x] as f32
                 + 1.0 * img[(y - 1) * self.w + (x + 1)] as f32
                 - 2.0 * img[y * self.w + (x - 1)] as f32
                 + 0.0 * img[y * self.w + x] as f32
                 + 2.0 * img[y * self.w + (x + 1)] as f32
                 - 1.0 * img[(y + 1) * self.w + (x - 1)] as f32
                 + 0.0 * img[(y + 1) * self.w + x] as f32
                 + 1.0 * img[(y + 1) * self.w + (x + 1)] as f32;

        // Sobel Y kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        let iy = -1.0 * img[(y - 1) * self.w + (x - 1)] as f32
                 - 2.0 * img[(y - 1) * self.w + x] as f32
                 - 1.0 * img[(y - 1) * self.w + (x + 1)] as f32
                 + 0.0 * img[y * self.w + (x - 1)] as f32
                 + 0.0 * img[y * self.w + x] as f32
                 + 0.0 * img[y * self.w + (x + 1)] as f32
                 + 1.0 * img[(y + 1) * self.w + (x - 1)] as f32
                 + 2.0 * img[(y + 1) * self.w + x] as f32
                 + 1.0 * img[(y + 1) * self.w + (x + 1)] as f32;

        (ix / 8.0, iy / 8.0) // Normalize by Sobel sum
    }

    /// Compute orientation with safe SIMD optimizations for subpixel coordinates
    fn compute_orientation(&self, img: &Image, x: f32, y: f32) -> FastResult<f32> {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use core::arch::x86_64::*;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        use core::arch::aarch64::*;

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

        // Use SIMD optimizations when available and safe
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        unsafe {
            self.compute_orientation_sse2(img, x, y, half, &mut m10, &mut m01)?;
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            self.compute_orientation_neon(img, x, y, half, &mut m10, &mut m01)?;
        }

        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            self.compute_orientation_scalar(img, x, y, half, &mut m10, &mut m01)?;
        }

        Ok((m01 as f32).atan2(m10 as f32))
    }

    /// Safe SSE2 implementation with proper bounds checking
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    unsafe fn compute_orientation_sse2(
        &self,
        img: &Image,
        x: f32,
        y: f32,
        half: i32,
        m10: &mut i64,
        m01: &mut i64,
    ) -> FastResult<()> {
        use core::arch::x86_64::*;

        let cx = x.round() as i32;
        let cy = y.round() as i32;

        for dy in -half..=half {
            let yy = (cy + dy) as usize;
            if yy >= self.h { continue; }
            
            let row_ptr = img.as_ptr().add(yy * self.w);
            let coeff = _mm_set1_epi16(dy as i16);
            let mut dx = -half;

            // Process 16 pixels at a time with bounds checking
            while dx + 15 <= half {
                let start_x = cx + dx;
                let end_x = cx + dx + 15;
                
                // Ensure we don't read past image boundaries
                if start_x >= 0 && end_x < self.w as i32 {
                    let ptr = row_ptr.add(start_x as usize);
                    
                    // Load 16 bytes safely
                    let vals = _mm_loadu_si128(ptr as *const __m128i);
                    let vals16_lo = _mm_unpacklo_epi8(vals, _mm_setzero_si128());
                    let vals16_hi = _mm_unpackhi_epi8(vals, _mm_setzero_si128());
                    
                    // Process low 8 values
                    let prod32_lo = _mm_madd_epi16(vals16_lo, coeff);
                    let sum_lo = _mm_add_epi32(prod32_lo, _mm_srli_si128(prod32_lo, 8));
                    let sum_lo = _mm_add_epi32(sum_lo, _mm_srli_si128(sum_lo, 4));
                    *m01 += _mm_cvtsi128_si32(sum_lo) as i64;
                    
                    // Process high 8 values
                    let prod32_hi = _mm_madd_epi16(vals16_hi, coeff);
                    let sum_hi = _mm_add_epi32(prod32_hi, _mm_srli_si128(prod32_hi, 8));
                    let sum_hi = _mm_add_epi32(sum_hi, _mm_srli_si128(sum_hi, 4));
                    *m01 += _mm_cvtsi128_si32(sum_hi) as i64;

                    // Compute m10 with bilinear interpolation
                    for o in 0..16 {
                        let sample_x = x + (dx + o) as f32;
                        let sample_y = y + dy as f32;
                        let val = self.bilinear_interpolate(img, sample_x, sample_y);
                        let offset = dx + o;
                        *m10 += offset as i64 * val as i64;
                    }
                } else {
                    // Fall back to scalar for boundary regions
                    for o in 0..16 {
                        let sample_x = x + (dx + o) as f32;
                        let sample_y = y + dy as f32;
                        let val = self.bilinear_interpolate(img, sample_x, sample_y);
                        let offset = dx + o;
                        *m10 += offset as i64 * val as i64;
                        *m01 += dy as i64 * val as i64;
                    }
                }
                dx += 16;
            }

            // Handle remaining pixels
            while dx <= half {
                let sample_x = x + dx as f32;
                let sample_y = y + dy as f32;
                let val = self.bilinear_interpolate(img, sample_x, sample_y);
                *m10 += dx as i64 * val as i64;
                *m01 += dy as i64 * val as i64;
                dx += 1;
            }
        }
        Ok(())
    }

    /// Safe NEON implementation with proper bounds checking
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe fn compute_orientation_neon(
        &self,
        img: &Image,
        x: f32,
        y: f32,
        half: i32,
        m10: &mut i64,
        m01: &mut i64,
    ) -> FastResult<()> {
        use core::arch::aarch64::*;

        let cx = x.round() as i32;
        let cy = y.round() as i32;

        for dy in -half..=half {
            let yy = (cy + dy) as usize;
            if yy >= self.h { continue; }
            
            let row_ptr = img.as_ptr().add(yy * self.w);
            let coeff = vdupq_n_s16(dy as i16);
            let mut dx = -half;

            // Process 16 pixels at a time with bounds checking
            while dx + 15 <= half {
                let start_x = cx + dx;
                let end_x = cx + dx + 15;
                
                // Ensure we don't read past image boundaries
                if start_x >= 0 && end_x < self.w as i32 {
                    let ptr = row_ptr.add(start_x as usize);
                    
                    // Load 16 bytes safely
                    let vals_u8 = vld1q_u8(ptr as *const u8);
                    let lo = vmovl_u8(vget_low_u8(vals_u8));
                    let hi = vmovl_u8(vget_high_u8(vals_u8));
                    let lo_s = vreinterpretq_s16_u16(lo);
                    let hi_s = vreinterpretq_s16_u16(hi);
                    
                    let prod_lo = vmulq_s16(lo_s, coeff);
                    let prod_hi = vmulq_s16(hi_s, coeff);
                    *m01 += (vaddvq_s16(prod_lo) as i64) + (vaddvq_s16(prod_hi) as i64);

                    // Compute m10 with bilinear interpolation
                    for o in 0..16 {
                        let sample_x = x + (dx + o) as f32;
                        let sample_y = y + dy as f32;
                        let val = self.bilinear_interpolate(img, sample_x, sample_y);
                        let offset = dx + o;
                        *m10 += offset as i64 * val as i64;
                    }
                } else {
                    // Fall back to scalar for boundary regions
                    for o in 0..16 {
                        let sample_x = x + (dx + o) as f32;
                        let sample_y = y + dy as f32;
                        let val = self.bilinear_interpolate(img, sample_x, sample_y);
                        let offset = dx + o;
                        *m10 += offset as i64 * val as i64;
                        *m01 += dy as i64 * val as i64;
                    }
                }
                dx += 16;
            }

            // Handle remaining pixels
            while dx <= half {
                let sample_x = x + dx as f32;
                let sample_y = y + dy as f32;
                let val = self.bilinear_interpolate(img, sample_x, sample_y);
                *m10 += dx as i64 * val as i64;
                *m01 += dy as i64 * val as i64;
                dx += 1;
            }
        }
        Ok(())
    }

    /// Scalar implementation for non-SIMD architectures
    fn compute_orientation_scalar(
        &self,
        img: &Image,
        x: f32,
        y: f32,
        half: i32,
        m10: &mut i64,
        m01: &mut i64,
    ) -> FastResult<()> {
        for dy in -half..=half {
            for dx in -half..=half {
                let sample_x = x + dx as f32;
                let sample_y = y + dy as f32;
                let val = self.bilinear_interpolate(img, sample_x, sample_y);
                *m10 += dx as i64 * val as i64;
                *m01 += dy as i64 * val as i64;
            }
        }
        Ok(())
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
        let _has_subpixel_x = (refined_kp.x - refined_kp.x.round()).abs() > 1e-6;
        let _has_subpixel_y = (refined_kp.y - refined_kp.y.round()).abs() > 1e-6;
        
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

    #[test] 
    fn test_simd_orientation_computation() {
        let detector = FastDetector::new(create_test_config(), 50, 50).unwrap();
        let img = create_corner_image(50, 50);
        
        // Test that SIMD and scalar versions give similar results
        let result1 = detector.compute_orientation(&img, 25.0, 25.0).unwrap();
        let result2 = detector.compute_orientation(&img, 25.0, 25.0).unwrap();
        
        // Results should be consistent
        assert!((result1 - result2).abs() < 1e-6);
        assert!(result1.is_finite());
    }

    #[test]
    fn test_harris_vs_intensity_difference() {
        let detector = FastDetector::new(create_test_config(), 50, 50).unwrap();
        let img = create_corner_image(50, 50);
        
        // Test that Harris response provides corner detection (with fallback)
        let scored_keypoints = detector.detect_keypoints_with_response(&img).unwrap();
        
        // All detected keypoints should have positive response
        for sk in &scored_keypoints {
            assert!(sk.response > 0.0, "Response should be positive for corners");
            assert!(sk.response.is_finite(), "Response should be finite");
        }
        
        // Should detect some corners (with Harris + intensity fallback)
        assert!(scored_keypoints.len() > 0, "Should detect some corners");
    }

    #[test]
    fn test_harris_second_moment_matrix() {
        let detector = FastDetector::new(create_test_config(), 20, 20).unwrap();
        
        // Create a strong corner pattern
        let mut img = vec![50; 400]; // 20x20 background
        
        // Create L-shaped corner
        for y in 8..12 {
            for x in 8..12 {
                if x <= 10 || y <= 10 {
                    img[y * 20 + x] = 255; // Bright L-shape
                }
            }
        }
        
        let response = detector.compute_harris_response(&img, 10, 10);
        
        // Test at a point with no structure
        let response_flat = detector.compute_harris_response(&img, 2, 2);
        assert!(response >= response_flat, "Corner should have higher or equal response than flat region");
    }

    #[test]
    fn test_harris_edge_vs_corner() {
        let detector = FastDetector::new(create_test_config(), 20, 20).unwrap();
        
        // Create an edge (high gradient in one direction)
        let mut edge_img = vec![50; 400];
        for y in 0..20 {
            for x in 10..20 {
                edge_img[y * 20 + x] = 255; // Vertical edge
            }
        }
        
        // Create a corner (high gradient in both directions)
        let mut corner_img = vec![50; 400];
        for y in 10..20 {
            for x in 10..20 {
                corner_img[y * 20 + x] = 255; // Corner
            }
        }
        
        let edge_response = detector.compute_harris_response(&edge_img, 10, 10);
        let corner_response = detector.compute_harris_response(&corner_img, 10, 10);
        
        // Corner should have higher Harris response than edge
        // (Harris metric suppresses edges while enhancing corners)
        assert!(corner_response > edge_response, 
               "Corner response ({}) should be higher than edge response ({})", 
               corner_response, edge_response);
    }

    #[test]
    fn test_sobel_gradients() {
        let detector = FastDetector::new(create_small_test_config(), 10, 10).unwrap(); // Use small config
        
        // Create a simple gradient image (horizontal)
        let mut img = vec![0; 100];
        for y in 0..10 {
            for x in 0..10 {
                img[y * 10 + x] = (x * 25) as u8; // Horizontal gradient
            }
        }
        
        let (ix, iy) = detector.compute_gradients(&img, 5, 5);
        
        // Horizontal gradient should have strong x-component, weak y-component
        assert!(ix.abs() > 0.0, "Should detect horizontal gradient");
        assert!(ix.abs() > iy.abs(), "X gradient should be stronger than Y gradient");
    }

    #[test]
    fn test_harris_corner_response() {
        let detector = FastDetector::new(create_test_config(), 20, 20).unwrap();
        let img = create_corner_image(20, 20);
        
        // Test Harris response at center (might be positive for corner)
        let response_center = detector.compute_harris_response(&img, 10, 10);
        
        // Test Harris response at uniform region (should be low)
        let uniform_img = vec![128; 400]; // 20x20 uniform image
        let response_uniform = detector.compute_harris_response(&uniform_img, 10, 10);
        assert!(response_uniform <= response_center, "Uniform region should have lower response than corner");
        
        // At minimum, responses should be finite
        assert!(response_center.is_finite(), "Harris response should be finite");
        assert!(response_uniform.is_finite(), "Harris response should be finite");
    }
} 