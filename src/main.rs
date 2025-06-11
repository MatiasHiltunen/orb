use rayon::prelude::*;

/// Row-major 8-bit grayscale image
pub type Image = Vec<u8>;

/// Key-point ≙ FAST corner + orientation (radians)
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: usize,
    pub y: usize,
    pub angle: f32,
}

/// 256-bit binary descriptor = 32 bytes
pub type Descriptor = [u8; 32];

#[derive(Debug, Clone)]
pub struct OrbConfig {
    pub threshold: u8,
    pub patch_size: usize,
    pub n_threads: usize,
}

impl Default for OrbConfig {
    fn default() -> Self {
        Self {
            threshold: 20,
            patch_size: 15,
            n_threads: num_cpus::get().max(1),
        }
    }
}

pub struct OrbMax {
    cfg: OrbConfig,
    w: usize,
    h: usize,
}

impl OrbMax {
    pub fn new(cfg: OrbConfig, width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0);
        assert!(cfg.patch_size & 1 == 1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.n_threads)
            .build_global()
            .ok();
        Self {
            cfg,
            w: width,
            h: height,
        }
    }

    fn detect_keypoints(&self, img: &Image) -> Vec<Keypoint> {
        const OFF: [(i32, i32); 16] = [
            (-3, 0),
            (-3, 1),
            (-2, 2),
            (-1, 3),
            (0, 3),
            (1, 3),
            (2, 2),
            (3, 1),
            (3, 0),
            (3, -1),
            (2, -2),
            (1, -3),
            (0, -3),
            (-1, -3),
            (-2, -2),
            (-3, -1),
        ];

        let rows = 3..self.h.saturating_sub(3);
        rows.into_par_iter()
            .flat_map_iter(|y| {
                let mut v = Vec::new();
                for x in 3..self.w - 3 {
                    let p = img[y * self.w + x];
                    let mut bri = 0;
                    let mut drk = 0;
                    for &(dx, dy) in &OFF {
                        // let xx = (x as i32 + dx) as usize;
                        // let yy = (y as i32 + dy) as usize;
                        //
                        let xx = (x as i32 + dx).clamp(0, (self.w - 1) as i32) as usize;
                        let yy = (y as i32 + dy).clamp(0, (self.h - 1) as i32) as usize;
                        let q = img[yy * self.w + xx];
                        // let q = img[yy * self.w + xx];
                        if q >= p.saturating_add(self.cfg.threshold) {
                            bri += 1;
                            if bri >= 12 {
                                let mut kp = Keypoint { x, y, angle: 0.0 };
                                self.compute_orientation(img, &mut kp);
                                v.push(kp);
                                break;
                            }
                        // } else if q + self.cfg.threshold <= p {
                        } else if q.saturating_add(self.cfg.threshold) <= p {
                            drk += 1;
                            if drk >= 12 {
                                let mut kp = Keypoint { x, y, angle: 0.0 };
                                self.compute_orientation(img, &mut kp);
                                v.push(kp);
                                break;
                            }
                        }
                    }
                }
                v
            })
            .collect()
    }

    #[inline(always)]
    fn compute_orientation(&self, img: &Image, kp: &mut Keypoint) {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use core::arch::x86_64::*;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        use core::arch::aarch64::*;

        let half = (self.cfg.patch_size / 2) as i32;
        let (cx, cy) = (kp.x as i32, kp.y as i32);
        let mut m10 = 0i32;
        let mut m01 = 0i32;

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        unsafe {
            for dy in -half..=half {
                let yy = (cy + dy).clamp(0, self.h as i32 - 1) as usize;
                let row = img.as_ptr().add(yy * self.w);
                let coeff = _mm_set1_epi16(dy as i16);
                let mut dx = -half;

                while dx + 15 <= half {
                    let base = (cx + dx).clamp(0, (self.w - 16) as i32) as usize;
                    let ptr = row.add(base);
                    let vals = _mm_loadu_si128(ptr as *const __m128i);
                    let vals16 = _mm_unpacklo_epi8(vals, _mm_setzero_si128());
                    let prod32 = _mm_madd_epi16(vals16, coeff);
                    let v = _mm_add_epi32(prod32, _mm_srli_si128(prod32, 8));
                    let v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
                    m01 += _mm_cvtsi128_si32(v);

                    for o in 0..16 {
                        let val = *ptr.add(o as usize) as i32;
                        let offset = dx.wrapping_add(o as i32);
                        m10 = m10.wrapping_add(offset.wrapping_mul(val));
                    }
                    dx += 16;
                }

                for dx in dx..=half {
                    let xx = (cx + dx).clamp(0, self.w as i32 - 1) as usize;
                    let val = img[yy * self.w + xx] as i32;
                    m10 += dx * val;
                    m01 += dy * val;
                }
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            for dy in -half..=half {
                let yy = (cy + dy).clamp(0, self.h as i32 - 1) as usize;
                let row = img.as_ptr().add(yy * self.w);
                let coeff = vdupq_n_s16(dy as i16);
                let mut dx = -half;

                while dx + 15 <= half {
                    let base = (cx + dx).clamp(0, (self.w - 16) as i32) as usize;
                    let ptr = row.add(base);
                    let vals_u8 = vld1q_u8(ptr as *const u8);
                    let lo = vmovl_u8(vget_low_u8(vals_u8));
                    let hi = vmovl_u8(vget_high_u8(vals_u8));
                    let lo_s = vreinterpretq_s16_u16(lo);
                    let hi_s = vreinterpretq_s16_u16(hi);
                    let prod_lo = vmulq_s16(lo_s, coeff);
                    let prod_hi = vmulq_s16(hi_s, coeff);
                    m01 += vaddvq_s16(prod_lo) as i32 + vaddvq_s16(prod_hi) as i32;

                    for o in 0..16 {
                        let val = *ptr.add(o as usize) as i32;
                        let offset = dx.wrapping_add(o as i32);
                        m10 = m10.wrapping_add(offset.wrapping_mul(val));
                    }
                    dx += 16;
                }

                for dx in dx..=half {
                    let xx = (cx + dx).clamp(0, self.w as i32 - 1) as usize;
                    let val = img[yy * self.w + xx] as i32;
                    m10 += dx * val;
                    m01 += dy * val;
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            for dy in -half..=half {
                let yy = (cy + dy).clamp(0, self.h as i32 - 1) as usize;
                for dx in -half..=half {
                    let xx = (cx + dx).clamp(0, self.w as i32 - 1) as usize;
                    let val = img[yy * self.w + xx] as i32;
                    m10 += dx * val;
                    m01 += dy * val;
                }
            }
        }

        kp.angle = (m01 as f32).atan2(m10 as f32);
    }

    fn generate_descriptors(&self, img: &Image, kps: &[Keypoint]) -> Vec<Descriptor> {
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

    pub fn detect_and_describe(&self, img: &Image) -> (Vec<Keypoint>, Vec<Descriptor>) {
        let kps = self.detect_keypoints(img);
        let desc = self.generate_descriptors(img, &kps);
        (kps, desc)
    }
}

// fn main() {
//     let img = ImageReader::open("lenna.png")
//         .expect("Failed to open image")
//         .decode()
//         .expect("Failed to decode image")
//         .to_luma8();

//     let (w, h) = img.dimensions();
//     let orb = OrbMax::new(OrbConfig::default(), w as usize, h as usize);
//     let t0 = std::time::Instant::now();
//     let (kps, desc) = orb.detect_and_describe(img.as_raw());

//     println!("Time taken: {:.2?}", t0.elapsed());
//     println!("Detected {} keypoints", kps.len());
//     println!("Generated {} descriptors", desc.len());
// }
// use image::Reader as ImageReader;
use image::{ImageBuffer, ImageReader, Rgba, RgbaImage};
use imageproc::drawing::draw_hollow_circle_mut;
use std::time::Instant;

fn main() {
    // Load grayscale image
    let img = ImageReader::open("lenna.png")
        .expect("Image not found")
        .decode()
        .expect("Decode failed")
        .to_luma8();

    let (w, h) = img.dimensions();
    let orb = OrbMax::new(OrbConfig::default(), w as usize, h as usize);

    // Time the full pipeline
    let t0 = Instant::now();
    let (kps, desc) = orb.detect_and_describe(img.as_raw());
    let elapsed = t0.elapsed();

    println!("Time taken: {:.2?}", elapsed);
    println!("Detected {} keypoints", kps.len());
    println!("Generated {} descriptors", desc.len());

    // Convert image to RGBA for drawing
    let mut output: RgbaImage = image::DynamicImage::ImageLuma8(img).into_rgba8();

    // Draw red circles at each keypoint
    for kp in &kps {
        draw_hollow_circle_mut(
            &mut output,
            (kp.x as i32, kp.y as i32),
            3,
            Rgba([255, 0, 0, 255]),
        );
    }

    // Save result
    output
        .save("lenna_keypoints.png")
        .expect("Failed to save output image");
    println!("Saved result image as lenna_keypoints.png");
}
