// Module declarations
pub mod error;
pub mod types;
pub mod pyramid;
pub mod corner_detection;
pub mod refinement;
pub mod preprocessing;
pub mod detector;
pub mod config;

#[cfg(feature = "simd")]
pub mod simd;

// Re-export public API
pub use error::{FastError, FastResult};
pub use types::{ScoredKeypoint, ScaleLevel};
pub use detector::FastDetector;
pub use config::{DetectorConfig, DetectorBuilder, ConfiguredDetector};

// Tests are included in individual modules 