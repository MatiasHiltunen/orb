use orb_core::Image;
use crate::error::FastResult;
use crate::types::ScaleLevel;

/// Image pyramid operations for multi-scale feature detection
pub struct ImagePyramid;

impl ImagePyramid {
    /// Generate scale levels for image pyramid
    pub fn generate_scale_levels(width: usize, height: usize) -> Vec<ScaleLevel> {
        let mut levels = Vec::new();
        let scale_factor = 1.2f32; // Standard ORB scale factor
        let mut current_scale = 1.0f32;
        let mut level = 0;

        loop {
            let scaled_width = ((width as f32) / current_scale) as usize;
            let scaled_height = ((height as f32) / current_scale) as usize;

            // Stop when image becomes too small for meaningful detection
            if scaled_width < 32 || scaled_height < 32 {
                break;
            }

            levels.push(ScaleLevel {
                level,
                scale: current_scale,
                width: scaled_width,
                height: scaled_height,
            });

            current_scale *= scale_factor;
            level += 1;

            // Limit to reasonable number of levels
            if level >= 8 {
                break;
            }
        }

        levels
    }

    /// Build image pyramid from base image
    pub fn build_image_pyramid(img: &Image, width: usize, height: usize, scale_levels: &[ScaleLevel]) -> FastResult<Vec<Image>> {
        let mut pyramid = Vec::with_capacity(scale_levels.len());
        
        for scale_level in scale_levels {
            if scale_level.level == 0 {
                // Base level - use original image
                pyramid.push(img.clone());
            } else {
                // Downsample for higher levels
                let downsampled = Self::downsample_image(img, width, height, scale_level.width, scale_level.height)?;
                pyramid.push(downsampled);
            }
        }
        
        Ok(pyramid)
    }

    /// Downsample image using bilinear interpolation
    fn downsample_image(img: &Image, src_width: usize, src_height: usize, target_width: usize, target_height: usize) -> FastResult<Image> {
        let mut downsampled = vec![0u8; target_width * target_height];
        
        let x_ratio = src_width as f32 / target_width as f32;
        let y_ratio = src_height as f32 / target_height as f32;
        
        for y in 0..target_height {
            for x in 0..target_width {
                let src_x = x as f32 * x_ratio;
                let src_y = y as f32 * y_ratio;
                
                let value = Self::bilinear_sample(img, src_width, src_height, src_x, src_y);
                downsampled[y * target_width + x] = value as u8;
            }
        }
        
        Ok(downsampled)
    }

    /// Sample image at fractional coordinates using bilinear interpolation
    fn bilinear_sample(img: &Image, width: usize, height: usize, x: f32, y: f32) -> f32 {
        let x1 = x.floor() as usize;
        let y1 = y.floor() as usize;
        let x2 = (x1 + 1).min(width - 1);
        let y2 = (y1 + 1).min(height - 1);
        
        let fx = x - x1 as f32;
        let fy = y - y1 as f32;
        
        let p11 = img[y1 * width + x1] as f32;
        let p12 = img[y1 * width + x2] as f32;
        let p21 = img[y2 * width + x1] as f32;
        let p22 = img[y2 * width + x2] as f32;
        
        let interpolated_top = p11 * (1.0 - fx) + p12 * fx;
        let interpolated_bottom = p21 * (1.0 - fx) + p22 * fx;
        
        interpolated_top * (1.0 - fy) + interpolated_bottom * fy
    }
} 