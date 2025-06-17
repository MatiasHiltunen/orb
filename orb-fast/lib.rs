// Module declarations
pub mod error;
pub mod types;
pub mod pyramid;
pub mod corner_detection;
pub mod refinement;
pub mod preprocessing;
pub mod detector;
pub mod config;
pub mod utils;
pub mod builder;
pub mod configured_detector;

#[cfg(feature = "simd")]
pub mod simd;

// Re-export public API
pub use error::{FastError, FastResult};
pub use types::{ScoredKeypoint, ScaleLevel};
pub use detector::FastDetector;
pub use config::DetectorConfig;
pub use builder::DetectorBuilder;
pub use configured_detector::ConfiguredDetector;

// Tests are included in individual modules 