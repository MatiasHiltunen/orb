use orb_core::{Image, Keypoint, Descriptor, OrbConfig, init_thread_pool};
use orb_fast::FastDetector;
use orb_brief::BriefGenerator;

pub use orb_core::{self, Image as OrbImage, Keypoint as OrbKeypoint, Descriptor as OrbDescriptor, OrbConfig as Config};

/// High-level ORB feature detector that combines FAST corner detection with BRIEF descriptors
pub struct OrbMax {
    fast_detector: FastDetector,
    brief_generator: BriefGenerator,
}

impl OrbMax {
    /// Create a new ORB detector with the given configuration and image dimensions
    pub fn new(cfg: OrbConfig, width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0);
        assert!(cfg.patch_size & 1 == 1);
        
        // Initialize thread pool
        init_thread_pool(cfg.n_threads).ok();
        
        Self {
            fast_detector: FastDetector::new(cfg, width, height),
            brief_generator: BriefGenerator::new(width, height),
        }
    }

    /// Detect keypoints using FAST corner detection
    pub fn detect_keypoints(&self, img: &Image) -> Vec<Keypoint> {
        self.fast_detector.detect_keypoints(img)
    }

    /// Generate BRIEF descriptors for the given keypoints
    pub fn generate_descriptors(&self, img: &Image, kps: &[Keypoint]) -> Vec<Descriptor> {
        self.brief_generator.generate_descriptors(img, kps)
    }

    /// Detect keypoints and generate descriptors in one step
    pub fn detect_and_describe(&self, img: &Image) -> (Vec<Keypoint>, Vec<Descriptor>) {
        let kps = self.detect_keypoints(img);
        let desc = self.generate_descriptors(img, &kps);
        (kps, desc)
    }
} 