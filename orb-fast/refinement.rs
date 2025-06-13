use orb_core::{Image, Keypoint};
use crate::error::{FastError, FastResult};
use crate::types::{ScoredKeypoint, ScaleLevel};

/// Subpixel refinement and orientation computation
pub struct KeypointRefinement;

impl KeypointRefinement {
    /// Refine keypoint to subpixel accuracy using quadratic surface fitting
    pub fn refine_keypoint_subpixel(img: &Image, width: usize, height: usize, kp: Keypoint) -> FastResult<Keypoint> {
        let x = kp.x as usize;
        let y = kp.y as usize;
        
        // Ensure we have enough border for 3x3 sampling
        if x < 1 || y < 1 || x >= width - 1 || y >= height - 1 {
            return Ok(kp);
        }
        
        // Sample 3x3 neighborhood for quadratic surface fitting
        let samples = [
            [Self::sample_safe(img, width, height, x - 1, y - 1), Self::sample_safe(img, width, height, x, y - 1), Self::sample_safe(img, width, height, x + 1, y - 1)],
            [Self::sample_safe(img, width, height, x - 1, y), Self::sample_safe(img, width, height, x, y), Self::sample_safe(img, width, height, x + 1, y)],
            [Self::sample_safe(img, width, height, x - 1, y + 1), Self::sample_safe(img, width, height, x, y + 1), Self::sample_safe(img, width, height, x + 1, y + 1)],
        ];
        
        // Fit quadratic surface: f(x,y) = Ax² + By² + Cxy + Dx + Ey + F
        // Use finite differences to find peak
        let dx = (samples[1][2] - samples[1][0]) / 2.0;
        let dy = (samples[2][1] - samples[0][1]) / 2.0;
        let dxx = samples[1][2] - 2.0 * samples[1][1] + samples[1][0];
        let dyy = samples[2][1] - 2.0 * samples[1][1] + samples[0][1];
        let dxy = (samples[2][2] - samples[2][0] - samples[0][2] + samples[0][0]) / 4.0;
        
        // Compute Hessian determinant for peak detection
        let det = dxx * dyy - dxy * dxy;
        
        if det.abs() < 1e-6 {
            // Degenerate case - return original point
            return Ok(kp);
        }
        
        // Find sub-pixel offset using Newton's method
        let offset_x = -(dyy * dx - dxy * dy) / det;
        let offset_y = -(dxx * dy - dxy * dx) / det;
        
        // Clamp offsets to reasonable range
        let offset_x = offset_x.clamp(-0.5, 0.5);
        let offset_y = offset_y.clamp(-0.5, 0.5);
        
        // Apply subpixel refinement
        let refined_x = kp.x + offset_x;
        let refined_y = kp.y + offset_y;
        
        Ok(Keypoint {
            x: refined_x,
            y: refined_y,
            angle: kp.angle,
        })
    }

    /// Safe image sampling with bounds checking
    fn sample_safe(img: &Image, width: usize, height: usize, x: usize, y: usize) -> f32 {
        if x < width && y < height && y * width + x < img.len() {
            img[y * width + x] as f32
        } else {
            0.0
        }
    }

    /// Compute orientation for keypoint using intensity centroid method
    pub fn compute_orientation(img: &Image, width: usize, height: usize, x: f32, y: f32, patch_size: usize) -> FastResult<f32> {
        let half = (patch_size / 2) as i32;
        let mut m10 = 0i64;
        let mut m01 = 0i64;
        
        // Choose SIMD implementation based on target architecture and features
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        unsafe {
            crate::simd::SimdOperations::compute_orientation_sse2(img, width, height, x, y, half, &mut m10, &mut m01)?;
        }
        
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            crate::simd::SimdOperations::compute_orientation_neon(img, width, height, x, y, half, &mut m10, &mut m01)?;
        }
        
        #[cfg(not(feature = "simd"))]
        {
            crate::simd::SimdOperations::compute_orientation_scalar(img, width, height, x, y, half, &mut m10, &mut m01)?;
        }
        
        // Compute orientation angle
        if m10 == 0 && m01 == 0 {
            Ok(0.0)
        } else {
            Ok((m01 as f32).atan2(m10 as f32))
        }
    }

    /// Compute orientation at specific scale level
    pub fn compute_orientation_at_scale(
        img: &Image,
        x: f32,
        y: f32,
        scale_level: &ScaleLevel,
        patch_size: usize,
    ) -> FastResult<f32> {
        let scaled_patch_size = ((patch_size as f32) / scale_level.scale) as usize;
        let effective_patch_size = scaled_patch_size.max(7); // Minimum patch size
        
        Self::compute_orientation(img, scale_level.width, scale_level.height, x, y, effective_patch_size)
    }

    /// Bilinear interpolation for fractional coordinates
    pub fn bilinear_interpolate(img: &Image, width: usize, height: usize, x: f32, y: f32) -> f32 {
        let x1 = x.floor() as usize;
        let y1 = y.floor() as usize;
        let x2 = (x1 + 1).min(width - 1);
        let y2 = (y1 + 1).min(height - 1);
        
        if x1 >= width || y1 >= height {
            return 0.0;
        }
        
        let fx = x - x1 as f32;
        let fy = y - y1 as f32;
        
        let p11 = Self::sample_safe(img, width, height, x1, y1);
        let p12 = Self::sample_safe(img, width, height, x2, y1);
        let p21 = Self::sample_safe(img, width, height, x1, y2);
        let p22 = Self::sample_safe(img, width, height, x2, y2);
        
        let interpolated_top = p11 * (1.0 - fx) + p12 * fx;
        let interpolated_bottom = p21 * (1.0 - fx) + p22 * fx;
        
        interpolated_top * (1.0 - fy) + interpolated_bottom * fy
    }

    /// Bilinear interpolation with scale level adjustment
    pub fn bilinear_interpolate_scaled(
        img: &Image,
        x: f32,
        y: f32,
        scale_level: &ScaleLevel,
    ) -> f32 {
        Self::bilinear_interpolate(img, scale_level.width, scale_level.height, x, y)
    }

    /// Non-maximum suppression to reduce duplicate keypoints
    pub fn non_maximum_suppression(keypoints: &[ScoredKeypoint], min_distance: f32) -> Vec<ScoredKeypoint> {
        if keypoints.is_empty() {
            return Vec::new();
        }
        
        let mut sorted_keypoints = keypoints.to_vec();
        sorted_keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        
        let mut suppressed: Vec<ScoredKeypoint> = Vec::new();
        let min_distance_sq = min_distance * min_distance;
        
        for candidate in sorted_keypoints {
            let mut is_local_max = true;
            
            for existing in &suppressed {
                let dx = candidate.keypoint.x - existing.keypoint.x;
                let dy = candidate.keypoint.y - existing.keypoint.y;
                let distance_sq = dx * dx + dy * dy;
                
                if distance_sq < min_distance_sq {
                    is_local_max = false;
                    break;
                }
            }
            
            if is_local_max {
                suppressed.push(candidate);
            }
        }
        
        suppressed
    }
} 