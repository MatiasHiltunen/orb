use orb_core::{Image, Keypoint, OrbConfig};
use crate::error::{FastError, FastResult};
use crate::types::{ScoredKeypoint, ScaleLevel};
use crate::pyramid::ImagePyramid;
use crate::corner_detection::CornerDetector;
use crate::refinement::KeypointRefinement;
use crate::preprocessing::ImagePreprocessing;
use rayon::prelude::*;

/// Main FAST corner detector with multi-scale capability
pub struct FastDetector {
    cfg: OrbConfig,
    w: usize,
    h: usize,
    scale_levels: Vec<ScaleLevel>,
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

        // Generate scale levels for pyramid
        let scale_levels = ImagePyramid::generate_scale_levels(width, height);

        Ok(Self {
            cfg,
            w: width,
            h: height,
            scale_levels,
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

    /// Detect keypoints with multi-scale detection
    pub fn detect_keypoints(&self, img: &Image) -> FastResult<Vec<Keypoint>> {
        let scored_keypoints = self.detect_keypoints_with_response(img)?;
        let suppressed = KeypointRefinement::non_maximum_suppression(&scored_keypoints, 3.0);
        
        // Apply subpixel refinement to the filtered keypoints
        let refined_keypoints = suppressed
            .into_iter()
            .filter_map(|sk| KeypointRefinement::refine_keypoint_subpixel(img, self.w, self.h, sk.keypoint).ok())
            .collect();
            
        Ok(refined_keypoints)
    }

    /// Detect keypoints across multiple scales with response scores
    pub fn detect_keypoints_with_response(&self, img: &Image) -> FastResult<Vec<ScoredKeypoint>> {
        // Validate input
        self.validate_image(img)?;

        // Build image pyramid
        let pyramid = ImagePyramid::build_image_pyramid(img, self.w, self.h, &self.scale_levels)?;
        
        // Detect keypoints at each scale level
        let scale_keypoints: Vec<Vec<ScoredKeypoint>> = self.scale_levels.iter()
            .zip(pyramid.iter())
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(scale_level, scaled_img)| {
                match self.detect_keypoints_at_scale(scaled_img, scale_level) {
                    Ok(keypoints) => keypoints,
                    Err(_) => Vec::new(), // Skip failed scales
                }
            })
            .collect();

        // Combine keypoints from all scales
        let mut all_keypoints: Vec<ScoredKeypoint> = scale_keypoints.into_iter().flatten().collect();

        // Transform coordinates back to original scale
        for _keypoint in all_keypoints.iter_mut() {
            // Note: Scale information is stored in ScaleLevel, not Keypoint
            // For now, coordinates are already in the correct scale
        }

        Ok(all_keypoints)
    }

    /// Detect keypoints at a specific scale level
    pub fn detect_keypoints_at_scale(&self, img: &Image, scale_level: &ScaleLevel) -> FastResult<Vec<ScoredKeypoint>> {
        // Compute adaptive thresholds for this scale
        let adaptive_thresholds = ImagePreprocessing::compute_adaptive_thresholds(
            img, 
            scale_level.width, 
            scale_level.height, 
            scale_level, 
            cfg!(feature = "rolling-window-stats")
        )?;

        // Detect corners using FAST algorithm
        let use_optimized_layout = cfg!(feature = "optimized-memory-layout");
        let mut keypoints = CornerDetector::detect_keypoints_at_scale(
            img, 
            scale_level, 
            &adaptive_thresholds, 
            use_optimized_layout
        )?;

        // Compute orientations for detected keypoints
        for keypoint in keypoints.iter_mut() {
            keypoint.keypoint.angle = KeypointRefinement::compute_orientation_at_scale(
                img,
                keypoint.keypoint.x,
                keypoint.keypoint.y,
                scale_level,
                self.cfg.patch_size,
            )?;
        }

        Ok(keypoints)
    }

    /// Get scale levels for this detector
    pub fn get_scale_levels(&self) -> &[ScaleLevel] {
        &self.scale_levels
    }

    /// Get detector configuration
    pub fn config(&self) -> &OrbConfig {
        &self.cfg
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Get adaptive threshold map for debugging
    pub fn get_adaptive_threshold_map(&self, img: &Image) -> FastResult<Vec<Vec<u8>>> {
        self.validate_image(img)?;
        
        // Use first scale level for threshold computation
        let scale_level = &self.scale_levels[0];
        ImagePreprocessing::compute_adaptive_thresholds(
            img, 
            self.w, 
            self.h, 
            scale_level, 
            cfg!(feature = "rolling-window-stats")
        )
    }

    /// Apply CLAHE preprocessing
    pub fn apply_clahe_preprocessing(&self, img: &Image) -> FastResult<Image> {
        self.validate_image(img)?;
        ImagePreprocessing::apply_clahe_preprocessing(img, self.w, self.h)
    }

    /// Compute Harris corner response at a specific location
    pub fn compute_harris_response(&self, img: &Image, x: usize, y: usize) -> f32 {
        CornerDetector::compute_harris_response(img, self.w, self.h, x, y)
    }

    /// Compute orientation using intensity centroid method
    pub fn compute_orientation(&self, img: &Image, x: f32, y: f32) -> FastResult<f32> {
        KeypointRefinement::compute_orientation(img, self.w, self.h, x, y, self.cfg.patch_size)
    }

    /// Refine keypoint to subpixel accuracy
    pub fn refine_keypoint_subpixel(&self, img: &Image, kp: Keypoint) -> FastResult<Keypoint> {
        self.validate_image(img)?;
        KeypointRefinement::refine_keypoint_subpixel(img, self.w, self.h, kp)
    }

    /// Apply non-maximum suppression
    pub fn non_maximum_suppression(&self, keypoints: &[ScoredKeypoint], min_distance: f32) -> Vec<ScoredKeypoint> {
        KeypointRefinement::non_maximum_suppression(keypoints, min_distance)
    }
} 