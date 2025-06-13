use orb_core::Keypoint;

/// Keypoint with corner response score for NMS
#[derive(Debug, Clone, Copy)]
pub struct ScoredKeypoint {
    pub keypoint: Keypoint,
    pub response: f32,
}

/// Scale information for pyramid levels
#[derive(Debug, Clone, Copy)]
pub struct ScaleLevel {
    pub level: usize,
    pub scale: f32,
    pub width: usize,
    pub height: usize,
}

/// Corner type classification for optimized processing
#[derive(Clone, Copy, Debug)]
pub(crate) enum CornerType {
    Bright,
    Dark,
    None,
} 