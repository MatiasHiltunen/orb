use orb_core::Image;
use crate::error::{FastError, FastResult};
use crate::types::ScaleLevel;

/// Image preprocessing algorithms (CLAHE, adaptive thresholding)
pub struct ImagePreprocessing;

impl ImagePreprocessing {
    /// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
    pub fn apply_clahe_preprocessing(img: &Image, width: usize, height: usize) -> FastResult<Image> {
        let tile_size = 8; // 8x8 tiles as per standard CLAHE
        let clip_limit = 2.0; // Standard contrast limiting factor
        
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        
        // Compute histograms for each tile
        let histograms = Self::compute_tile_histograms(img, width, height, tile_size, tiles_x, tiles_y);
        
        // Apply contrast limiting
        let limited_histograms = Self::apply_contrast_limiting(&histograms, clip_limit);
        
        // Compute cumulative distribution functions
        let tile_cdfs = Self::compute_tile_cdfs(&limited_histograms);
        
        // Apply adaptive equalization
        Self::apply_adaptive_equalization(img, width, height, &tile_cdfs, tile_size, tiles_x, tiles_y)
    }

    /// Compute histograms for each tile
    fn compute_tile_histograms(
        img: &Image,
        width: usize,
        height: usize,
        tile_size: usize,
        tiles_x: usize,
        tiles_y: usize,
    ) -> Vec<Vec<u32>> {
        let mut histograms = Vec::with_capacity(tiles_x * tiles_y);
        
        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let mut histogram = vec![0u32; 256];
                
                let start_x = tile_x * tile_size;
                let end_x = ((tile_x + 1) * tile_size).min(width);
                let start_y = tile_y * tile_size;
                let end_y = ((tile_y + 1) * tile_size).min(height);
                
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        if y * width + x < img.len() {
                            let pixel = img[y * width + x] as usize;
                            histogram[pixel] += 1;
                        }
                    }
                }
                
                histograms.push(histogram);
            }
        }
        
        histograms
    }

    /// Apply contrast limiting to histograms
    fn apply_contrast_limiting(histograms: &[Vec<u32>], clip_limit: f32) -> Vec<Vec<u32>> {
        histograms.iter().map(|histogram| {
            let total_pixels: u32 = histogram.iter().sum();
            let avg_height = total_pixels as f32 / 256.0;
            let clip_threshold = (avg_height * clip_limit) as u32;
            
            let mut limited = histogram.clone();
            let mut excess = 0u32;
            
            // Clip histogram and collect excess
            for count in limited.iter_mut() {
                if *count > clip_threshold {
                    excess += *count - clip_threshold;
                    *count = clip_threshold;
                }
            }
            
            // Redistribute excess uniformly
            let redistribution = excess / 256;
            let remainder = excess % 256;
            
            for (i, count) in limited.iter_mut().enumerate() {
                *count += redistribution;
                if i < remainder as usize {
                    *count += 1;
                }
            }
            
            limited
        }).collect()
    }

    /// Compute cumulative distribution functions for tiles
    fn compute_tile_cdfs(histograms: &[Vec<u32>]) -> Vec<Vec<f32>> {
        histograms.iter().map(|histogram| {
            let total_pixels: u32 = histogram.iter().sum();
            
            if total_pixels == 0 {
                return vec![0.0; 256];
            }
            
            let mut cdf = Vec::with_capacity(256);
            let mut cumulative = 0u32;
            
            for &count in histogram {
                cumulative += count;
                cdf.push((cumulative as f32 / total_pixels as f32) * 255.0);
            }
            
            cdf
        }).collect()
    }

    /// Apply adaptive equalization using tile CDFs
    fn apply_adaptive_equalization(
        img: &Image,
        width: usize,
        height: usize,
        tile_cdfs: &[Vec<f32>],
        tile_size: usize,
        tiles_x: usize,
        tiles_y: usize,
    ) -> FastResult<Image> {
        let mut equalized = vec![0u8; img.len()];
        
        for y in 0..height {
            for x in 0..width {
                if y * width + x >= img.len() {
                    continue;
                }
                
                let pixel = img[y * width + x];
                
                // Find which tile this pixel belongs to
                let tile_x = (x / tile_size).min(tiles_x - 1);
                let tile_y = (y / tile_size).min(tiles_y - 1);
                
                let tile_idx = tile_y * tiles_x + tile_x;
                
                // Use current tile's CDF (simplified - no interpolation for now)
                if tile_idx < tile_cdfs.len() && (pixel as usize) < tile_cdfs[tile_idx].len() {
                    equalized[y * width + x] = tile_cdfs[tile_idx][pixel as usize].clamp(0.0, 255.0) as u8;
                } else {
                    equalized[y * width + x] = pixel;
                }
            }
        }
        
        Ok(equalized)
    }

    /// Compute adaptive thresholds based on local image statistics
    pub fn compute_adaptive_thresholds(
        img: &Image,
        width: usize,
        height: usize,
        scale_level: &ScaleLevel,
        _use_rolling_window: bool,
    ) -> FastResult<Vec<Vec<u8>>> {
        Self::compute_adaptive_thresholds_basic(img, width, height, scale_level)
    }

    /// Basic adaptive threshold computation using block statistics
    fn compute_adaptive_thresholds_basic(
        img: &Image,
        width: usize,
        height: usize,
        scale_level: &ScaleLevel,
    ) -> FastResult<Vec<Vec<u8>>> {
        let block_size = 16;
        let blocks_x = (scale_level.width + block_size - 1) / block_size;
        let blocks_y = (scale_level.height + block_size - 1) / block_size;
        
        let mut thresholds = vec![vec![10u8; blocks_x]; blocks_y];
        
        for block_y in 0..blocks_y {
            for block_x in 0..blocks_x {
                let start_x = block_x * block_size;
                let end_x = ((block_x + 1) * block_size).min(scale_level.width);
                let start_y = block_y * block_size;
                let end_y = ((block_y + 1) * block_size).min(scale_level.height);
                
                let mut sum = 0u64;
                let mut sum_sq = 0u64;
                let mut count = 0;
                
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        if y * width + x < img.len() {
                            let pixel = img[y * width + x] as u64;
                            sum += pixel;
                            sum_sq += pixel * pixel;
                            count += 1;
                        }
                    }
                }
                
                if count > 0 {
                    let mean = sum as f64 / count as f64;
                    let variance = (sum_sq as f64 / count as f64) - (mean * mean);
                    let std_dev = variance.sqrt();
                    
                    // Adaptive threshold based on local statistics
                    let base_threshold = 10.0;
                    let contrast_factor = Self::compute_contrast_factor(std_dev);
                    let adaptive_threshold = (base_threshold * contrast_factor).clamp(5.0, 50.0);
                    
                    thresholds[block_y][block_x] = adaptive_threshold as u8;
                } else {
                    thresholds[block_y][block_x] = 10;
                }
            }
        }
        
        Ok(thresholds)
    }

    /// Compute contrast factor for adaptive thresholding
    fn compute_contrast_factor(std_dev: f64) -> f64 {
        // Sigmoid function to map std_dev to threshold multiplier
        let normalized = std_dev / 50.0; // Normalize to typical image std_dev range
        1.0 + 2.0 / (1.0 + (-2.0 * (normalized - 0.5)).exp())
    }
} 