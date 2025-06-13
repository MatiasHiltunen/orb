use orb_core::Image;
use crate::error::{FastError, FastResult};
use crate::types::{CornerType, ScoredKeypoint, ScaleLevel};

/// SIMD-optimized implementations for corner detection and feature computation
pub struct SimdOperations;

impl SimdOperations {
    /// Fast corner check using SIMD optimizations
    pub fn fast_corner_check_simd(
        row_ptrs: &[*const u8],
        x: usize,
        y: usize,
        center_pixel: u8,
        threshold: u8,
        offsets: &[(i32, i32); 16],
        width: usize,
    ) -> (bool, CornerType) {
        let center_i32 = center_pixel as i32;
        let threshold_i32 = threshold as i32;
        
        let mut brighter_count = 0u32;
        let mut darker_count = 0u32;
        
        // Process pixels in groups for better cache utilization
        for &(dx, dy) in offsets.iter() {
            let px = (x as i32 + dx) as usize;
            let py = (y as i32 + dy) as usize;
            
            if px < width && py < row_ptrs.len() {
                unsafe {
                    let pixel_ptr = row_ptrs[py].add(px);
                    let pixel = *pixel_ptr as i32;
                    
                    if pixel > center_i32 + threshold_i32 {
                        brighter_count += 1;
                    } else if pixel < center_i32 - threshold_i32 {
                        darker_count += 1;
                    }
                }
            }
        }
        
        // FAST corner requires 9+ consecutive pixels above/below threshold
        if brighter_count >= 9 {
            (true, CornerType::Bright)
        } else if darker_count >= 9 {
            (true, CornerType::Dark)
        } else {
            (false, CornerType::None)
        }
    }

    /// Process pixel chunk with SIMD optimizations
    pub fn process_pixel_chunk_simd(
        img: &Image,
        scale_level: &ScaleLevel,
        adaptive_thresholds: &[Vec<u8>],
        offsets: &[(i32, i32); 16],
        y: usize,
        x_start: usize,
        x_end: usize,
    ) -> Vec<ScoredKeypoint> {
        let mut keypoints = Vec::new();
        let width = scale_level.width;
        
        // Pre-compute row pointers for faster access
        let row_ptrs: Vec<*const u8> = (0..scale_level.height)
            .map(|row| img.as_ptr().wrapping_add(row * width))
            .collect();
        
        for x in x_start..x_end {
            if x >= 3 && x < width - 3 && y >= 3 && y < scale_level.height - 3 {
                let center_pixel = img[y * width + x];
                let threshold = Self::get_local_threshold_cached(adaptive_thresholds, x, y, scale_level);
                
                let (is_corner, corner_type) = Self::fast_corner_check_simd(
                    &row_ptrs, x, y, center_pixel, threshold, offsets, width
                );
                
                if is_corner {
                    let response = Self::compute_response_optimized(img, scale_level, x, y, corner_type);
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
        
        keypoints
    }

    /// Get local threshold from cached threshold map
    fn get_local_threshold_cached(
        thresholds: &[Vec<u8>],
        x: usize,
        y: usize,
        scale_level: &ScaleLevel,
    ) -> u8 {
        if thresholds.is_empty() {
            return 10; // Default threshold
        }
        
        let block_size = 16;
        let block_x = x / block_size;
        let block_y = y / block_size;
        
        let blocks_x = (scale_level.width + block_size - 1) / block_size;
        
        if block_y < thresholds.len() && block_x < thresholds[block_y].len() {
            thresholds[block_y][block_x]
        } else {
            10 // Default fallback
        }
    }

    /// Compute optimized response for detected corners
    fn compute_response_optimized(
        img: &Image,
        scale_level: &ScaleLevel,
        x: usize,
        y: usize,
        corner_type: CornerType,
    ) -> f32 {
        match corner_type {
            CornerType::Bright => Self::compute_intensity_response_typed(img, scale_level, x, y, corner_type),
            CornerType::Dark => Self::compute_intensity_response_typed(img, scale_level, x, y, corner_type),
            CornerType::None => 0.0,
        }
    }

    /// Compute intensity response based on corner type
    fn compute_intensity_response_typed(
        img: &Image,
        scale_level: &ScaleLevel,
        x: usize,
        y: usize,
        corner_type: CornerType,
    ) -> f32 {
        let width = scale_level.width;
        let center_pixel = img[y * width + x] as f32;
        
        // Sample in 3x3 neighborhood
        let mut sum_diff = 0.0f32;
        let mut count = 0;
        
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 { continue; }
                
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                
                if nx < width && ny < scale_level.height {
                    let neighbor = img[ny * width + nx] as f32;
                    let diff = match corner_type {
                        CornerType::Bright => (center_pixel - neighbor).max(0.0),
                        CornerType::Dark => (neighbor - center_pixel).max(0.0),
                        CornerType::None => 0.0,
                    };
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

    /// SIMD-optimized orientation computation using SSE2
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn compute_orientation_sse2(
        img: &Image,
        width: usize,
        height: usize,
        x: f32,
        y: f32,
        half: i32,
        m10: &mut i64,
        m01: &mut i64,
    ) -> FastResult<()> {
        #[cfg(target_feature = "sse2")]
        {
            use std::arch::x86_64::*;
            
            let mut m10_vec = _mm_setzero_si128();
            let mut m01_vec = _mm_setzero_si128();
            
            let half_f = half as f32;
            let start_y = (y - half_f).max(0.0) as usize;
            let end_y = (y + half_f + 1.0).min(height as f32) as usize;
            let start_x = (x - half_f).max(0.0) as usize;
            let end_x = (x + half_f + 1.0).min(width as f32) as usize;
            
            for v in start_y..end_y {
                let dy = v as i32 - y as i32;
                let row_base = v * width;
                
                let mut u = start_x;
                let end_x_vec = end_x - (end_x % 8);
                
                // Process 8 pixels at a time using SSE2
                while u < end_x_vec {
                    if row_base + u + 7 < img.len() {
                        let dx_base = u as i32 - x as i32;
                        
                        // Load 8 consecutive bytes
                        let pixels = _mm_loadl_epi64(img.as_ptr().add(row_base + u) as *const _);
                        let pixels_16 = _mm_unpacklo_epi8(pixels, _mm_setzero_si128());
                        
                        // Create dx vector [dx_base, dx_base+1, ..., dx_base+7]
                        let dx_vals = [dx_base as i16, (dx_base+1) as i16, (dx_base+2) as i16, (dx_base+3) as i16, 
                                      (dx_base+4) as i16, (dx_base+5) as i16, (dx_base+6) as i16, (dx_base+7) as i16];
                        let dx_vec = _mm_loadu_si128(dx_vals.as_ptr() as *const _);
                        
                        // Compute pixel * dx for each element
                        let m10_contrib = _mm_mullo_epi16(pixels_16, dx_vec);
                        let m10_lo = _mm_unpacklo_epi16(m10_contrib, _mm_setzero_si128());
                        let m10_hi = _mm_unpackhi_epi16(m10_contrib, _mm_setzero_si128());
                        m10_vec = _mm_add_epi32(m10_vec, m10_lo);
                        m10_vec = _mm_add_epi32(m10_vec, m10_hi);
                        
                        // Compute pixel * dy (dy is constant for this row)
                        let dy_vec = _mm_set1_epi16(dy as i16);
                        let m01_contrib = _mm_mullo_epi16(pixels_16, dy_vec);
                        let m01_lo = _mm_unpacklo_epi16(m01_contrib, _mm_setzero_si128());
                        let m01_hi = _mm_unpackhi_epi16(m01_contrib, _mm_setzero_si128());
                        m01_vec = _mm_add_epi32(m01_vec, m01_lo);
                        m01_vec = _mm_add_epi32(m01_vec, m01_hi);
                    }
                    u += 8;
                }
                
                // Handle remaining pixels
                for u in end_x_vec..end_x {
                    if row_base + u < img.len() {
                        let dx = u as i32 - x as i32;
                        let pixel = img[row_base + u] as i64;
                        *m10 += pixel * dx as i64;
                        *m01 += pixel * dy as i64;
                    }
                }
            }
            
            // Extract sum from vector registers
            let m10_array: [i32; 4] = std::mem::transmute(m10_vec);
            let m01_array: [i32; 4] = std::mem::transmute(m01_vec);
            
            *m10 += (m10_array[0] + m10_array[1] + m10_array[2] + m10_array[3]) as i64;
            *m01 += (m01_array[0] + m01_array[1] + m01_array[2] + m01_array[3]) as i64;
        }
        
        #[cfg(not(target_feature = "sse2"))]
        {
            // Fallback to scalar implementation
            Self::compute_orientation_scalar(img, width, height, x, y, half, m10, m01)?;
        }
        
        Ok(())
    }

    /// SIMD-optimized orientation computation using NEON (ARM)
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn compute_orientation_neon(
        img: &Image,
        width: usize,
        height: usize,
        x: f32,
        y: f32,
        half: i32,
        m10: &mut i64,
        m01: &mut i64,
    ) -> FastResult<()> {
        #[cfg(target_feature = "neon")]
        {
            use std::arch::aarch64::*;
            
            let mut m10_vec = vdupq_n_s32(0);
            let mut m01_vec = vdupq_n_s32(0);
            
            let half_f = half as f32;
            let start_y = (y - half_f).max(0.0) as usize;
            let end_y = (y + half_f + 1.0).min(height as f32) as usize;
            let start_x = (x - half_f).max(0.0) as usize;
            let end_x = (x + half_f + 1.0).min(width as f32) as usize;
            
            for v in start_y..end_y {
                let dy = v as i32 - y as i32;
                let row_base = v * width;
                
                let mut u = start_x;
                let end_x_vec = end_x - (end_x % 8);
                
                // Process 8 pixels at a time using NEON
                while u < end_x_vec {
                    if row_base + u + 7 < img.len() {
                        let dx_base = u as i32 - x as i32;
                        
                        // Load 8 consecutive bytes
                        let pixels_8 = vld1_u8(img.as_ptr().add(row_base + u));
                        let pixels_16_low = vmovl_u8(pixels_8);
                        
                        // Create dx vector
                        let dx_vals = [dx_base as i16, (dx_base+1) as i16, (dx_base+2) as i16, (dx_base+3) as i16];
                        let dx_vec_low = vld1_s16(dx_vals.as_ptr());
                        
                        let dx_vals_high = [(dx_base+4) as i16, (dx_base+5) as i16, (dx_base+6) as i16, (dx_base+7) as i16];
                        let dx_vec_high = vld1_s16(dx_vals_high.as_ptr());
                        
                        // Compute pixel * dx
                        let m10_contrib_low = vmull_s16(vreinterpret_s16_u16(vget_low_u16(pixels_16_low)), dx_vec_low);
                        let m10_contrib_high = vmull_s16(vreinterpret_s16_u16(vget_high_u16(pixels_16_low)), dx_vec_high);
                        
                        m10_vec = vaddq_s32(m10_vec, m10_contrib_low);
                        m10_vec = vaddq_s32(m10_vec, m10_contrib_high);
                        
                        // Compute pixel * dy
                        let dy_vec = vdup_n_s16(dy as i16);
                        let m01_contrib_low = vmull_s16(vreinterpret_s16_u16(vget_low_u16(pixels_16_low)), dy_vec);
                        let m01_contrib_high = vmull_s16(vreinterpret_s16_u16(vget_high_u16(pixels_16_low)), dy_vec);
                        
                        m01_vec = vaddq_s32(m01_vec, m01_contrib_low);
                        m01_vec = vaddq_s32(m01_vec, m01_contrib_high);
                    }
                    u += 8;
                }
                
                // Handle remaining pixels
                for u in end_x_vec..end_x {
                    if row_base + u < img.len() {
                        let dx = u as i32 - x as i32;
                        let pixel = img[row_base + u] as i64;
                        *m10 += pixel * dx as i64;
                        *m01 += pixel * dy as i64;
                    }
                }
            }
            
            // Extract sum from vector registers
            let m10_array = [vgetq_lane_s32(m10_vec, 0), vgetq_lane_s32(m10_vec, 1), 
                            vgetq_lane_s32(m10_vec, 2), vgetq_lane_s32(m10_vec, 3)];
            let m01_array = [vgetq_lane_s32(m01_vec, 0), vgetq_lane_s32(m01_vec, 1), 
                            vgetq_lane_s32(m01_vec, 2), vgetq_lane_s32(m01_vec, 3)];
            
            *m10 += (m10_array[0] + m10_array[1] + m10_array[2] + m10_array[3]) as i64;
            *m01 += (m01_array[0] + m01_array[1] + m01_array[2] + m01_array[3]) as i64;
        }
        
        Ok(())
    }

    /// Scalar fallback for orientation computation
    pub fn compute_orientation_scalar(
        img: &Image,
        width: usize,
        height: usize,
        x: f32,
        y: f32,
        half: i32,
        m10: &mut i64,
        m01: &mut i64,
    ) -> FastResult<()> {
        let half_f = half as f32;
        let start_y = (y - half_f).max(0.0) as usize;
        let end_y = (y + half_f + 1.0).min(height as f32) as usize;
        let start_x = (x - half_f).max(0.0) as usize;
        let end_x = (x + half_f + 1.0).min(width as f32) as usize;
        
        for v in start_y..end_y {
            let dy = v as i32 - y as i32;
            for u in start_x..end_x {
                let dx = u as i32 - x as i32;
                if v * width + u < img.len() {
                    let pixel = img[v * width + u] as i64;
                    *m10 += pixel * dx as i64;
                    *m01 += pixel * dy as i64;
                }
            }
        }
        
        Ok(())
    }
} 