use orb_core::{Image, Keypoint, Descriptor, OrbConfig, init_thread_pool};
use orb_fast::{FastDetector, FastError};
use orb_brief::BriefGenerator;

pub use orb_core::{self, Image as OrbImage, Keypoint as OrbKeypoint, Descriptor as OrbDescriptor, OrbConfig as Config};

#[derive(Debug)]
pub enum OrbError {
    Fast(FastError),
    ThreadPool(rayon::ThreadPoolBuildError),
}

impl std::fmt::Display for OrbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrbError::Fast(e) => write!(f, "FAST error: {}", e),
            OrbError::ThreadPool(e) => write!(f, "Thread pool error: {}", e),
        }
    }
}

impl std::error::Error for OrbError {}

impl From<FastError> for OrbError {
    fn from(err: FastError) -> Self {
        OrbError::Fast(err)
    }
}

impl From<rayon::ThreadPoolBuildError> for OrbError {
    fn from(err: rayon::ThreadPoolBuildError) -> Self {
        OrbError::ThreadPool(err)
    }
}

pub type OrbResult<T> = Result<T, OrbError>;

/// High-level ORB feature detector that combines FAST corner detection with BRIEF descriptors
pub struct OrbMax {
    fast_detector: FastDetector,
    brief_generator: BriefGenerator,
}

impl OrbMax {
    /// Create a new ORB detector with the given configuration and image dimensions
    pub fn new(cfg: OrbConfig, width: usize, height: usize) -> OrbResult<Self> {
        // Initialize thread pool
        init_thread_pool(cfg.n_threads)?;
        
        let fast_detector = FastDetector::new(cfg, width, height)?;
        let brief_generator = BriefGenerator::new(width, height);
        
        Ok(Self {
            fast_detector,
            brief_generator,
        })
    }

    /// Detect keypoints using FAST corner detection
    pub fn detect_keypoints(&self, img: &Image) -> OrbResult<Vec<Keypoint>> {
        Ok(self.fast_detector.detect_keypoints(img)?)
    }

    /// Generate BRIEF descriptors for the given keypoints
    pub fn generate_descriptors(&self, img: &Image, kps: &[Keypoint]) -> Vec<Descriptor> {
        self.brief_generator.generate_descriptors(img, kps)
    }

    /// Detect keypoints and generate descriptors in one step
    pub fn detect_and_describe(&self, img: &Image) -> OrbResult<(Vec<Keypoint>, Vec<Descriptor>)> {
        let kps = self.detect_keypoints(img)?;
        let desc = self.generate_descriptors(img, &kps);
        Ok((kps, desc))
    }

    /// Get detector configuration
    pub fn config(&self) -> &OrbConfig {
        self.fast_detector.config()
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        self.fast_detector.dimensions()
    }
} 