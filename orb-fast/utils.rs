/// Utility functions for corner detection algorithms

/// Check if there are at least `min_count` consecutive true values in the circular array
/// using a branch-free bitmask approach for better performance
pub fn has_consecutive_pixels_bitmask(pixels: &[bool; 16], min_count: usize) -> bool {
    if min_count > 16 || min_count == 0 {
        return false;
    }
    
    // Convert boolean array to 16-bit mask
    let mut mask: u16 = 0;
    for (i, &pixel) in pixels.iter().enumerate() {
        if pixel {
            mask |= 1 << i;
        }
    }
    
    // Check for consecutive runs using bit manipulation
    // For a run of length n, we need: mask & (mask << 1) & (mask << 2) & ... & (mask << (n-1))
    let mut test_mask = mask;
    for i in 1..min_count {
        // Rotate left with wrap-around for circular buffer
        let shifted = (mask << i) | (mask >> (16 - i));
        test_mask &= shifted;
        if test_mask == 0 {
            return false; // Early exit if no consecutive runs possible
        }
    }
    
    test_mask != 0
}

/// Fallback implementation for consecutive pixel checking
/// (kept for reference and edge cases)
pub fn has_consecutive_pixels_fallback(pixels: &[bool; 16], min_count: usize) -> bool {
    if min_count > 16 {
        return false;
    }
    
    let mut max_consecutive = 0;
    let mut current_consecutive = 0;
    
    // Check twice the array length to handle wrap-around
    for i in 0..(16 * 2) {
        let idx = i % 16;
        if pixels[idx] {
            current_consecutive += 1;
            max_consecutive = max_consecutive.max(current_consecutive);
            if max_consecutive >= min_count {
                return true;
            }
        } else {
            current_consecutive = 0;
        }
    }
    
    false
}

/// Main function for checking consecutive pixels
/// Uses optimized bitmask approach for better performance
pub fn has_consecutive_pixels(pixels: &[bool; 16], min_count: usize) -> bool {
    has_consecutive_pixels_bitmask(pixels, min_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consecutive_pixels_simple() {
        let mut pixels = [false; 16];
        // Set 9 consecutive pixels
        for i in 0..9 {
            pixels[i] = true;
        }
        assert!(has_consecutive_pixels(&pixels, 9));
        assert!(!has_consecutive_pixels(&pixels, 10));
    }

    #[test]
    fn test_consecutive_pixels_wrap_around() {
        let mut pixels = [false; 16];
        // Set pixels that wrap around the circular buffer
        for i in 12..16 {
            pixels[i] = true;
        }
        for i in 0..5 {
            pixels[i] = true;
        }
        assert!(has_consecutive_pixels(&pixels, 9));
    }

    #[test]
    fn test_non_consecutive_pixels() {
        let mut pixels = [false; 16];
        // Set alternating pixels
        for i in (0..16).step_by(2) {
            pixels[i] = true;
        }
        assert!(!has_consecutive_pixels(&pixels, 4));
    }

    #[test]
    fn test_bitmask_vs_fallback() {
        let test_cases = [
            ([true; 16], 9),
            ([false; 16], 1),
            ([true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false], 2),
        ];

        for (pixels, min_count) in test_cases {
            assert_eq!(
                has_consecutive_pixels_bitmask(&pixels, min_count),
                has_consecutive_pixels_fallback(&pixels, min_count),
                "Mismatch for min_count={}", min_count
            );
        }
    }
} 