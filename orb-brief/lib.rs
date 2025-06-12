use orb_core::{Image, Keypoint, Descriptor};
use rayon::prelude::*;

pub struct BriefGenerator {
    w: usize,
    h: usize,
}

impl BriefGenerator {
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0);
        Self { w: width, h: height }
    }

    pub fn generate_descriptors(&self, img: &Image, kps: &[Keypoint]) -> Vec<Descriptor> {
        const PAIRS: [(i32, i32, i32, i32); 32] = [
            (-4, 0, 4, 0),
            (-3, -3, 3, 3),
            (-2, -4, 2, 4),
            (0, -4, 0, 4),
            (1, -3, -1, 3),
            (2, -2, -2, 2),
            (3, -1, -3, 1),
            (4, 0, -4, 0),
            (-4, -2, 4, 2),
            (-3, 0, 3, 0),
            (-2, -2, 2, 2),
            (0, -3, 0, 3),
            (-4, -4, 4, 4),
            (-3, -2, 3, 2),
            (-2, 0, 2, 0),
            (0, -2, 0, 2),
            (-4, -3, 4, 3),
            (-3, -4, 3, 4),
            (-2, -3, 2, 3),
            (0, -1, 0, 1),
            (1, -2, -1, 2),
            (2, -1, -2, 1),
            (3, 0, -3, 0),
            (4, 1, -4, -1),
            (-1, -4, 1, 4),
            (-2, -3, 2, 3),
            (-3, -2, 3, 2),
            (-4, -1, 4, 1),
            (0, 1, 0, -1),
            (1, 2, -1, -2),
            (2, 3, -2, -3),
            (3, 4, -3, -4),
        ];

        kps.par_iter()
            .map(|kp| {
                let (s, c) = kp.angle.sin_cos();
                let (cx, cy) = (kp.x as f32, kp.y as f32);
                let mut d = [0u8; 32];
                for (i, &(dx1, dy1, dx2, dy2)) in PAIRS.iter().enumerate() {
                    let (rx1, ry1) = (
                        cx + c * dx1 as f32 - s * dy1 as f32,
                        cy + s * dx1 as f32 + c * dy1 as f32,
                    );
                    let (rx2, ry2) = (
                        cx + c * dx2 as f32 - s * dy2 as f32,
                        cy + s * dx2 as f32 + c * dy2 as f32,
                    );
                    let x1 = rx1.round().clamp(0.0, (self.w - 1) as f32) as usize;
                    let y1 = ry1.round().clamp(0.0, (self.h - 1) as f32) as usize;
                    let x2 = rx2.round().clamp(0.0, (self.w - 1) as f32) as usize;
                    let y2 = ry2.round().clamp(0.0, (self.h - 1) as f32) as usize;

                    let bit = (img[y1 * self.w + x1] < img[y2 * self.w + x2]) as u8;
                    d[i >> 3] |= bit << (7 - (i & 7));
                }
                d
            })
            .collect()
    }
} 