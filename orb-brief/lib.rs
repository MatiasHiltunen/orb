use orb_core::{Image, Keypoint, Descriptor};
use rayon::prelude::*;

const DESCRIPTOR_SIZE: usize = 32;

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
        // TODO: This should be 256 pairs for a 32-byte descriptor for optimal performance.
        // These pairs can be learned from a dataset (see the BRIEF paper).
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
                let (cx, cy) = (kp.x, kp.y);
                let mut d = [0u8; DESCRIPTOR_SIZE];

                // With only 32 pairs, this will only fill the first 4 bytes of the descriptor.
                for (i, &(dx1, dy1, dx2, dy2)) in PAIRS.iter().enumerate() {
                    // Apply rotation and translation for subpixel coordinates
                    let (rx1, ry1) = (
                        cx + c * dx1 as f32 - s * dy1 as f32,
                        cy + s * dx1 as f32 + c * dy1 as f32,
                    );
                    let (rx2, ry2) = (
                        cx + c * dx2 as f32 - s * dy2 as f32,
                        cy + s * dx2 as f32 + c * dy2 as f32,
                    );

                    // Sample using bilinear interpolation for subpixel accuracy
                    let val1 = self.bilinear_sample(img, rx1, ry1);
                    let val2 = self.bilinear_sample(img, rx2, ry2);

                    let bit = (val1 < val2) as u8;
                    d[i / 8] |= bit << (i % 8);
                }
                d
            })
            .collect()
    }

    /// Bilinear interpolation for subpixel sampling
    fn bilinear_sample(&self, img: &Image, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let x1 = x0 + 1.0;
        let y1 = y0 + 1.0;

        // Handle boundary conditions
        if x0 < 0.0 || y0 < 0.0 || x1 >= self.w as f32 || y1 >= self.h as f32 {
            // Clamp to image bounds for boundary samples
            let cx = x.round().clamp(0.0, (self.w - 1) as f32) as usize;
            let cy = y.round().clamp(0.0, (self.h - 1) as f32) as usize;
            return img[cy * self.w + cx] as f32;
        }

        let dx = x - x0;
        let dy = y - y0;

        let x0_idx = x0 as usize;
        let y0_idx = y0 as usize;
        let x1_idx = (x1 as usize).min(self.w - 1);
        let y1_idx = (y1 as usize).min(self.h - 1);

        let p00 = img[y0_idx * self.w + x0_idx] as f32;
        let p10 = img[y0_idx * self.w + x1_idx] as f32;
        let p01 = img[y1_idx * self.w + x0_idx] as f32;
        let p11 = img[y1_idx * self.w + x1_idx] as f32;

        // Bilinear interpolation
        let top = p00 * (1.0 - dx) + p10 * dx;
        let bottom = p01 * (1.0 - dx) + p11 * dx;
        
        top * (1.0 - dy) + bottom * dy
    }
} 