use orb_core::Image;
use crate::error::{FastError, FastResult};
use crate::types::{CornerType, ScoredKeypoint, ScaleLevel};
use rayon::prelude::*;

/// Corner detection algorithms (FAST and Harris)
pub struct CornerDetector;

impl CornerDetector {
    /// FAST circle offsets for corner detection
    pub const FAST_OFFSETS: [(i32, i32); 16] = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3),
    ];

    /// Detect keypoints at a specific scale using parallel processing
    pub fn detect_keypoints_at_scale(
        img: &Image,
        scale_level: &ScaleLevel,
        adaptive_thresholds: &[Vec<u8>],
        use_optimized_layout: bool,
    ) -> FastResult<Vec<ScoredKeypoint>> {
        if use_optimized_layout {
            Self::detect_keypoints_optimized_layout(img, scale_level, adaptive_thresholds, &Self::FAST_OFFSETS)
        } else {
            Self::detect_keypoints_basic_layout(img, scale_level, adaptive_thresholds, &Self::FAST_OFFSETS)
        }
    }

    /// Detect keypoints using basic row-by-row layout
    fn detect_keypoints_basic_layout(
        img: &Image,
        scale_level: &ScaleLevel,
        adaptive_thresholds: &[Vec<u8>],
        offsets: &[(i32, i32); 16],
    ) -> FastResult<Vec<ScoredKeypoint>> {
        let mut keypoints = Vec::new();
        let width = scale_level.width;
        let height = scale_level.height;

        for y in 3..height-3 {
            for x in 3..width-3 {
                let center_pixel = img[y * width + x];
                let threshold = Self::get_local_threshold(adaptive_thresholds, x, y, scale_level);
                
                if Self::is_fast_corner(img, width, x, y, center_pixel, threshold, offsets) {
                    let response = Self::compute_intensity_response_basic(
                        img, scale_level, x, y, center_pixel, threshold, offsets
                    );
                    
                    keypoints.push(ScoredKeypoint {
                        keypoint: orb_core::Keypoint {
                            x: x as f32,
                            y: y as f32,
                            angle: 0.0,
                        },
                        response,
                    });
                }
            }
        }

        Ok(keypoints)
    }

    /// Detect keypoints using optimized chunked layout with parallel processing
    fn detect_keypoints_optimized_layout(
        img: &Image,
        scale_level: &ScaleLevel,
        adaptive_thresholds: &[Vec<u8>],
        offsets: &[(i32, i32); 16],
    ) -> FastResult<Vec<ScoredKeypoint>> {
        let width = scale_level.width;
        let height = scale_level.height;
        const CHUNK_SIZE: usize = 64;

        let all_keypoints: Vec<Vec<ScoredKeypoint>> = (3..height-3)
            .into_par_iter()
            .map(|y| {
                let mut row_keypoints = Vec::new();
                
                for x_start in (3..width-3).step_by(CHUNK_SIZE) {
                    let x_end = (x_start + CHUNK_SIZE).min(width - 3);
                    
                    #[cfg(feature = "simd")]
                    {
                        use crate::simd::SimdOperations;
                        let chunk_keypoints = SimdOperations::process_pixel_chunk_simd(
                            img, scale_level, adaptive_thresholds, offsets, y, x_start, x_end
                        );
                        row_keypoints.extend(chunk_keypoints);
                    }
                    
                    #[cfg(not(feature = "simd"))]
                    {
                        for x in x_start..x_end {
                            let center_pixel = img[y * width + x];
                            let threshold = Self::get_local_threshold(adaptive_thresholds, x, y, scale_level);
                            
                            if Self::is_fast_corner(img, width, x, y, center_pixel, threshold, offsets) {
                                let response = Self::compute_intensity_response_basic(
                                    img, scale_level, x, y, center_pixel, threshold, offsets
                                );
                                
                                                                 row_keypoints.push(ScoredKeypoint {
                                     keypoint: orb_core::Keypoint {
                                         x: x as f32,
                                         y: y as f32,
                                         angle: 0.0,
                                     },
                                     response,
                                 });
                            }
                        }
                    }
                }
                
                row_keypoints
            })
            .collect();

        let keypoints = all_keypoints.into_iter().flatten().collect();
        Ok(keypoints)
    }

    /// Check if a pixel is a FAST corner
    fn is_fast_corner(
        img: &Image,
        width: usize,
        x: usize,
        y: usize,
        center_pixel: u8,
        threshold: u8,
        offsets: &[(i32, i32); 16],
    ) -> bool {
        let center_i32 = center_pixel as i32;
        let threshold_i32 = threshold as i32;
        
        let mut brighter_count = 0;
        let mut darker_count = 0;
        
        for &(dx, dy) in offsets.iter() {
            let px = (x as i32 + dx) as usize;
            let py = (y as i32 + dy) as usize;
            
            if px < width && py * width + px < img.len() {
                let pixel = img[py * width + px] as i32;
                
                if pixel > center_i32 + threshold_i32 {
                    brighter_count += 1;
                } else if pixel < center_i32 - threshold_i32 {
                    darker_count += 1;
                }
            }
        }
        
        // FAST corner requires 9+ consecutive pixels above/below threshold
        brighter_count >= 9 || darker_count >= 9
    }

    /// Compute basic intensity response for corner
    fn compute_intensity_response_basic(
        img: &Image,
        scale_level: &ScaleLevel,
        x: usize,
        y: usize,
        center_pixel: u8,
        threshold: u8,
        offsets: &[(i32, i32); 16],
    ) -> f32 {
        let center_f32 = center_pixel as f32;
        let width = scale_level.width;
        let mut sum_diff = 0.0f32;
        let mut count = 0;
        
        for &(dx, dy) in offsets.iter() {
            let px = (x as i32 + dx) as usize;
            let py = (y as i32 + dy) as usize;
            
            if px < width && py * width + px < img.len() {
                let pixel = img[py * width + px] as f32;
                let diff = (center_f32 - pixel).abs();
                if diff > threshold as f32 {
                    sum_diff += diff * diff;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            sum_diff / count as f32
        } else {
            0.0
        }
    }

    /// Get local threshold for adaptive thresholding
    fn get_local_threshold(
        thresholds: &[Vec<u8>],
        x: usize,
        y: usize,
        _scale_level: &ScaleLevel,
    ) -> u8 {
        if thresholds.is_empty() {
            return 10; // Default threshold
        }
        
        let block_size = 16;
        let block_x = x / block_size;
        let block_y = y / block_size;
        
        if block_y < thresholds.len() && block_x < thresholds[block_y].len() {
            thresholds[block_y][block_x]
        } else {
            10 // Default fallback
        }
    }

    /// Compute Harris corner response at a pixel
    pub fn compute_harris_response(img: &Image, width: usize, height: usize, x: usize, y: usize) -> f32 {
        #[cfg(feature = "integral-images")]
        {
            Self::compute_harris_response_integral(img, width, height, x, y)
        }
        
        #[cfg(not(feature = "integral-images"))]
        {
            Self::compute_harris_response_direct(img, width, height, x, y)
        }
    }

    /// Direct Harris corner response computation
    fn compute_harris_response_direct(img: &Image, width: usize, height: usize, x: usize, y: usize) -> f32 {
        if x < 2 || y < 2 || x >= width - 2 || y >= height - 2 {
            return 0.0;
        }

        let mut ixx = 0.0f64;
        let mut ixy = 0.0f64;
        let mut iyy = 0.0f64;

        // 5x5 window for Harris computation
        for dy in -2..=2 {
            for dx in -2..=2 {
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                
                if nx < width - 1 && ny < height - 1 {
                    let (gx, gy) = Self::compute_gradients(img, width, nx, ny);
                    
                    ixx += (gx * gx) as f64;
                    ixy += (gx * gy) as f64;
                    iyy += (gy * gy) as f64;
                }
            }
        }

        // Harris corner response: det(M) - k * trace(M)^2
        let k = 0.04f64;
        let det = ixx * iyy - ixy * ixy;
        let trace = ixx + iyy;
        let harris_response = det - k * trace * trace;

        // Only return positive responses (corners)
        if harris_response > 0.0 {
            harris_response as f32
        } else {
            0.0
        }
    }

    /// Compute image gradients using Sobel operator
    fn compute_gradients(img: &Image, width: usize, x: usize, y: usize) -> (f32, f32) {
        if x == 0 || y == 0 || x >= width - 1 || y >= width {
            return (0.0, 0.0);
        }

        // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
        let gx = (img[(y - 1) * width + (x + 1)] as f32) * 1.0
            + (img[y * width + (x + 1)] as f32) * 2.0
            + (img[(y + 1) * width + (x + 1)] as f32) * 1.0
            - (img[(y - 1) * width + (x - 1)] as f32) * 1.0
            - (img[y * width + (x - 1)] as f32) * 2.0
            - (img[(y + 1) * width + (x - 1)] as f32) * 1.0;

        // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
        let gy = (img[(y + 1) * width + (x - 1)] as f32) * 1.0
            + (img[(y + 1) * width + x] as f32) * 2.0
            + (img[(y + 1) * width + (x + 1)] as f32) * 1.0
            - (img[(y - 1) * width + (x - 1)] as f32) * 1.0
            - (img[(y - 1) * width + x] as f32) * 2.0
            - (img[(y - 1) * width + (x + 1)] as f32) * 1.0;

        (gx / 8.0, gy / 8.0)
    }

    /// Integral image-based Harris response (when feature enabled)
    #[cfg(feature = "integral-images")]
    fn compute_harris_response_integral(img: &Image, width: usize, height: usize, x: usize, y: usize) -> f32 {
        // Implementation would go here for integral image optimization
        // For now, fallback to direct computation
        Self::compute_harris_response_direct(img, width, height, x, y)
    }
} 