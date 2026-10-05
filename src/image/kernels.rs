//! Fast integer convolution kernels shared by the Pillow and PyTorch resamplers.
//!
//! Both libraries compute every output sample as
//! `clamp((init + sum_k w[k] * src[first + k]) >> precision, 0, 255)` with exact integer
//! arithmetic (no intermediate overflow), so the evaluation order is free: we always convolve
//! along *rows of rows* (contiguous memory) and implement the horizontal pass as
//! transpose -> row convolution -> transpose, in strips of rows that stay in L2. Results are
//! bit-identical to the straightforward per-pixel loops (checked by the unit tests).
//!
//! Kernels:
//! * `i16` weights (PyTorch quantizes its weights to `i16`) on x86_64 with AVX2:
//!   `_mm256_madd_epi16` on pairs of taps, accumulators kept in registers for 32 output bytes
//!   at a time (runtime detection; scalar fallback elsewhere);
//! * `i32` weights (Pillow's 22-bit fixed point): auto-vectorized `i32` multiply-add, compiled
//!   for AVX2 when available.
//!
//! Attribution: the fixed-point convolution semantics follow Pillow's `Resample.c` (MIT-CMU /
//! HPND license) and PyTorch's `UpSampleKernelAVXAntialias.h` (BSD-3-Clause, itself derived
//! from Pillow-SIMD). See THIRD_PARTY_NOTICES.md.

/// Separable 1-D filter in fixed point.
pub(crate) struct FixedFilter<'a> {
    /// `(first input index, number of taps)` per output index.
    pub bounds: &'a [(usize, usize)],
    /// `ksize` weights per output index (only the first `taps` are used).
    pub weights: &'a [i32],
    /// The same weights as `i16` when they fit (enables the `madd` kernel on x86_64).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub w16: Option<&'a [i16]>,
    pub ksize: usize,
    pub precision: u32,
}

#[inline(always)]
fn accumulate_generic(acc: &mut [i32], line: &[u8], w: i32) {
    for (a, &p) in acc.iter_mut().zip(line.iter()) {
        *a = a.wrapping_add(p as i32 * w);
    }
}

#[inline(always)]
fn finish_generic(out: &mut [u8], acc: &[i32], precision: u32) {
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        *o = (a >> precision).clamp(0, 255) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve_avx2(src: &[u8], row_len: usize, f: &FixedFilter, out: &mut [u8]) {
    convolve_impl(src, row_len, f, out)
}

#[inline(always)]
fn convolve_impl(src: &[u8], row_len: usize, f: &FixedFilter, out: &mut [u8]) {
    let init = 1i32 << (f.precision - 1);
    let mut acc = vec![0i32; row_len];
    for (j, &(first, taps)) in f.bounds.iter().enumerate() {
        acc.iter_mut().for_each(|a| *a = init);
        let w = &f.weights[j * f.ksize..j * f.ksize + taps];
        for (k, &wk) in w.iter().enumerate() {
            let r = first + k;
            accumulate_generic(&mut acc, &src[r * row_len..(r + 1) * row_len], wk);
        }
        finish_generic(&mut out[j * row_len..(j + 1) * row_len], &acc, f.precision);
    }
}

/// `i16` weights, AVX2: for each output row and each block of 32 bytes, accumulate tap pairs
/// with `madd_epi16` in four `i32x8` registers, then shift, saturate and pack. The per-lane
/// interleaving of `unpack{lo,hi}` is undone exactly by the per-lane `packs`/`packus`, so the
/// 32 output bytes come out in order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY (whole body): the caller checked AVX2 and that every row `first + k` exists; vector
// loads/stores only cover columns `x..x + 32 <= row_len`.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn convolve_madd_avx2(src: &[u8], row_len: usize, f: &FixedFilter, w16: &[i16], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let init = 1i32 << (f.precision - 1);
    let shift = _mm_cvtsi32_si128(f.precision as i32);
    let zero = _mm256_setzero_si256();
    let base = src.as_ptr();
    let blocks = row_len / 32;
    for (j, &(first, taps)) in f.bounds.iter().enumerate() {
        let w = &w16[j * f.ksize..j * f.ksize + taps];
        // Weight pairs (w[k], w[k+1]) packed in i32 lanes; an odd last tap pairs with 0.
        let mut pairs = [0i32; 64];
        let n_pairs = taps.div_ceil(2);
        let mut pairs_vec;
        let pairs: &mut [i32] = if n_pairs <= 64 {
            &mut pairs[..n_pairs]
        } else {
            pairs_vec = vec![0i32; n_pairs];
            &mut pairs_vec
        };
        for (p, slot) in pairs.iter_mut().enumerate() {
            let lo = w[2 * p] as u16 as i32;
            let hi = if 2 * p + 1 < taps { w[2 * p + 1] as i32 } else { 0 };
            *slot = lo | (hi << 16);
        }
        let orow = out.as_mut_ptr().add(j * row_len);
        for b in 0..blocks {
            let x = b * 32;
            let mut a0 = _mm256_set1_epi32(init);
            let mut a1 = a0;
            let mut a2 = a0;
            let mut a3 = a0;
            for (p, &wp) in pairs.iter().enumerate() {
                let k = 2 * p;
                let r0 = base.add((first + k) * row_len + x);
                // For an odd last tap, re-read the same row (its weight is 0).
                let r1 = if k + 1 < taps { r0.add(row_len) } else { r0 };
                let v0 = _mm256_loadu_si256(r0 as *const __m256i);
                let v1 = _mm256_loadu_si256(r1 as *const __m256i);
                let wv = _mm256_set1_epi32(wp);
                let lo = _mm256_unpacklo_epi8(v0, v1);
                let hi = _mm256_unpackhi_epi8(v0, v1);
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_unpacklo_epi8(lo, zero), wv));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_unpackhi_epi8(lo, zero), wv));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_unpacklo_epi8(hi, zero), wv));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_unpackhi_epi8(hi, zero), wv));
            }
            a0 = _mm256_sra_epi32(a0, shift);
            a1 = _mm256_sra_epi32(a1, shift);
            a2 = _mm256_sra_epi32(a2, shift);
            a3 = _mm256_sra_epi32(a3, shift);
            let p01 = _mm256_packs_epi32(a0, a1);
            let p23 = _mm256_packs_epi32(a2, a3);
            let bytes = _mm256_packus_epi16(p01, p23);
            _mm256_storeu_si256(orow.add(x) as *mut __m256i, bytes);
        }
        // Tail columns.
        for x in blocks * 32..row_len {
            let mut s = init;
            for (k, &wk) in w.iter().enumerate() {
                s = s.wrapping_add(*base.add((first + k) * row_len + x) as i32 * wk as i32);
            }
            *orow.add(x) = (s >> f.precision).clamp(0, 255) as u8;
        }
    }
}

/// AVX2 available and not disabled with `HF_PROCESSORS_FORCE_SCALAR=1` (used to run the golden
/// suite on the portable path).
#[cfg(target_arch = "x86_64")]
fn use_avx2() -> bool {
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX2.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2")
            && std::env::var("HF_PROCESSORS_FORCE_SCALAR").map_or(true, |v| v.is_empty() || v == "0")
    })
}

/// Resample along the row axis into `out` (`f.bounds.len() * row_len` bytes): `src` holds rows
/// of `row_len` bytes.
pub(crate) fn convolve_rows_into(src: &[u8], row_len: usize, f: &FixedFilter, out: &mut [u8]) {
    assert_eq!(out.len(), f.bounds.len() * row_len);
    if let Some(&(first, taps)) = f.bounds.iter().max_by_key(|(a, n)| a + n) {
        assert!((first + taps) * row_len <= src.len(), "filter reads past the input");
    }
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2() {
            // SAFETY: AVX2 support was just checked; all row reads are in bounds (asserted
            // above; 32-byte loads only cover columns < row_len).
            unsafe {
                match f.w16 {
                    Some(w16) => convolve_madd_avx2(src, row_len, f, w16, out),
                    None => convolve_avx2(src, row_len, f, out),
                }
            }
            return;
        }
    }
    convolve_impl(src, row_len, f, out);
}

/// Resample along the row axis: `src` holds `n_rows` rows of `row_len` bytes; the result
/// holds `f.bounds.len()` rows.
pub(crate) fn convolve_rows(src: &[u8], row_len: usize, f: &FixedFilter) -> Vec<u8> {
    let mut out = vec![0u8; f.bounds.len() * row_len];
    convolve_rows_into(src, row_len, f, &mut out);
    out
}

/// Transpose `src` (`h` rows of `w` pixels of `C` bytes) into `out` (`w` rows of `h` pixels),
/// writing output row `x` at `out[x * out_stride + out_offset ..]`.
#[inline(always)]
fn transpose_c<const C: usize>(src: &[u8], w: usize, h: usize, out: &mut [u8], out_stride: usize, out_offset: usize) {
    let src: &[[u8; C]] = as_pixels(src);
    let out: &mut [[u8; C]] = as_pixels_mut(out);
    const B: usize = 64;
    for y0 in (0..h).step_by(B) {
        let y1 = (y0 + B).min(h);
        for x0 in (0..w).step_by(B) {
            let x1 = (x0 + B).min(w);
            for x in x0..x1 {
                let o = x * out_stride + out_offset;
                let dst = &mut out[o + y0..o + y1];
                for (d, y) in dst.iter_mut().zip(y0..y1) {
                    *d = src[y * w + x];
                }
            }
        }
    }
}

fn as_pixels<const C: usize>(s: &[u8]) -> &[[u8; C]] {
    assert_eq!(s.len() % C, 0);
    // SAFETY: [u8; C] has alignment 1 and size C; length checked above.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const [u8; C], s.len() / C) }
}

fn as_pixels_mut<const C: usize>(s: &mut [u8]) -> &mut [[u8; C]] {
    assert_eq!(s.len() % C, 0);
    // SAFETY: as above, and the borrow is unique.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut [u8; C], s.len() / C) }
}

/// Transpose an interleaved `h x w x c` image (`src`) into `out` (`w` rows), placing it at
/// pixel column `out_offset` of rows that are `out_stride` pixels long.
fn transpose_into(src: &[u8], w: usize, h: usize, c: usize, out: &mut [u8], out_stride: usize, out_offset: usize) {
    match c {
        1 => transpose_c::<1>(src, w, h, out, out_stride, out_offset),
        2 => transpose_c::<2>(src, w, h, out, out_stride, out_offset),
        3 => transpose_c::<3>(src, w, h, out, out_stride, out_offset),
        4 => transpose_c::<4>(src, w, h, out, out_stride, out_offset),
        _ => unreachable!("channels={c}"),
    }
}

/// Transpose an interleaved `h x w x c` image into `w x h x c`.
#[cfg(test)]
pub(crate) fn transpose(src: &[u8], w: usize, h: usize, c: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    transpose_into(src, w, h, c, &mut out, h, 0);
    out
}

/// Rows per strip of the horizontal pass: the transposed strip (`w * STRIP * c` bytes) stays in
/// L2 for typical widths, and `STRIP * c` is a multiple of 32 (full SIMD blocks).
const STRIP: usize = 32;

/// Horizontal pass over rows `[row0, row0 + n_rows)` of an interleaved image: for each strip
/// of rows, transpose -> convolve rows -> transpose back into the output.
pub(crate) fn horizontal(
    src: &[u8],
    w: usize,
    c: usize,
    row0: usize,
    n_rows: usize,
    f: &FixedFilter,
) -> Vec<u8> {
    let out_w = f.bounds.len();
    let mut out = vec![0u8; out_w * n_rows * c];
    let mut t = vec![0u8; w * STRIP.min(n_rows) * c];
    let mut r = vec![0u8; out_w * STRIP.min(n_rows) * c];
    let mut y = 0;
    while y < n_rows {
        let s = STRIP.min(n_rows - y);
        let rows = &src[(row0 + y) * w * c..(row0 + y + s) * w * c];
        let t = &mut t[..w * s * c];
        transpose_into(rows, w, s, c, t, s, 0); // w rows of s pixels
        let r = &mut r[..out_w * s * c];
        convolve_rows_into(t, s * c, f, r); // out_w rows of s pixels
        // r is (out_w x s): transpose into output rows y..y+s (pixel stride out_w).
        let dst = &mut out[y * out_w * c..(y + s) * out_w * c];
        transpose_into(r, s, out_w, c, dst, out_w, 0);
        y += s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    /// The madd kernel (i16 weights, odd/even tap counts, tails) equals the generic loop.
    #[test]
    fn madd_kernel_matches_generic() {
        for (row_len, n_in, n_out, seed) in [(96usize, 50usize, 7usize, 1u64), (1008, 37, 5, 2), (33, 9, 9, 3), (31, 4, 2, 4)] {
            let src = rnd_bytes(row_len * n_in, seed);
            let ksize = n_in.min(13);
            let mut bounds = Vec::new();
            let mut w16 = Vec::new();
            let raw = rnd_bytes(n_out * ksize * 2, seed + 10);
            for j in 0..n_out {
                let taps = 1 + (j * 5) % ksize;
                let first = (j * 3) % (n_in - taps + 1);
                bounds.push((first, taps));
                for k in 0..ksize {
                    // Mix of large positive and negative weights.
                    let v = i16::from_le_bytes([raw[2 * (j * ksize + k)], raw[2 * (j * ksize + k) + 1]]) / 2;
                    w16.push(v);
                }
            }
            let w32: Vec<i32> = w16.iter().map(|&v| v as i32).collect();
            for precision in [7u32, 14] {
                let f16 = FixedFilter { bounds: &bounds, weights: &w32, w16: Some(&w16), ksize, precision };
                let f32_ = FixedFilter { bounds: &bounds, weights: &w32, w16: None, ksize, precision };
                let mut want = vec![0u8; n_out * row_len];
                convolve_impl(&src, row_len, &f32_, &mut want);
                assert_eq!(convolve_rows(&src, row_len, &f16), want, "row_len {row_len} precision {precision}");
            }
        }
    }
}
