use orb_core::{Image, Keypoint, OrbConfig};
use rayon::prelude::*;

pub struct FastDetector {
    cfg: OrbConfig,
    w: usize,
    h: usize,
}

impl FastDetector {
    pub fn new(cfg: OrbConfig, width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0);
        assert!(cfg.patch_size & 1 == 1);
        Self {
            cfg,
            w: width,
            h: height,
        }
    }

    pub fn detect_keypoints(&self, img: &Image) -> Vec<Keypoint> {
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
                        let xx = (x as i32 + dx).clamp(0, (self.w - 1) as i32) as usize;
                        let yy = (y as i32 + dy).clamp(0, (self.h - 1) as i32) as usize;
                        let q = img[yy * self.w + xx];
                        if q >= p.saturating_add(self.cfg.threshold) {
                            bri += 1;
                            if bri >= 12 {
                                let mut kp = Keypoint { x, y, angle: 0.0 };
                                self.compute_orientation(img, &mut kp);
                                v.push(kp);
                                break;
                            }
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
    pub fn compute_orientation(&self, img: &Image, kp: &mut Keypoint) {
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
} 