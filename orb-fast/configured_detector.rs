use orb_core::{Image, Keypoint};
use crate::error::FastResult;
use crate::types::{ScoredKeypoint, ScaleLevel};
use crate::detector::FastDetector;
use crate::builder::DetectorBuilder;

/// A FAST detector that has been configured with a specific builder.
///
/// This struct holds a `FastDetector` and a `DetectorBuilder`,
/// providing a convenient way to run the detector and access its configuration.
#[derive(Debug, Clone)]
pub struct ConfiguredDetector {
    pub(crate) detector: FastDetector,
    pub(crate) config: DetectorBuilder,
}

impl ConfiguredDetector {
    /// Detect keypoints in the given image.
    ///
    /// # Arguments
    /// * `img` - A row-major 8-bit grayscale image.
    ///
    /// # Returns
    /// A `FastResult` containing a vector of `Keypoint`s.
    pub fn detect_keypoints(&self, img: &Image) -> FastResult<Vec<Keypoint>> {
        if img.len() != self.config.width() * self.config.height() {
            return Err(crate::error::FastError::InvalidImageSize {
                width: self.config.width(),
                height: self.config.height(),
            });
        }
        self.detector.detect_keypoints(img)
    }

    /// Detect keypoints with their corner response scores.
    ///
    /// # Arguments
    /// * `img` - A row-major 8-bit grayscale image.
    ///
    /// # Returns
    /// A `FastResult` containing a vector of `ScoredKeypoint`s.
    pub fn detect_keypoints_with_response(&self, img: &Image) -> FastResult<Vec<ScoredKeypoint>> {
        if img.len() != self.config.width() * self.config.height() {
            return Err(crate::error::FastError::InvalidImageSize {
                width: self.config.width(),
                height: self.config.height(),
            });
        }
        self.detector.detect_keypoints_with_response(img)
    }

    /// Get a reference to the underlying `FastDetector`.
    pub fn detector(&self) -> &FastDetector {
        &self.detector
    }

    /// Get a summary of the detector's configuration.
    pub fn config_summary(&self) -> String {
        self.config.summary()
    }

    /// Get the image dimensions (width, height) the detector is configured for.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.config.width(), self.config.height())
    }

    /// Get the scale levels of the image pyramid.
    pub fn scale_levels(&self) -> &[ScaleLevel] {
        self.detector.get_scale_levels()
    }
} 