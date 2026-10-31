//! Minimal deterministic PRNG (xorshift64*) + Gaussian sampler.
//! Zero-dependency so all results are reproducible from a single `--seed`.

#[derive(Clone)]
pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which is a fixed point for xorshift.
        Self { s: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform float in [0, 1), 24 bits of mantissa.
    pub fn uniform01(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform integer in [0, n).
    pub fn rand_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal via Box–Muller. Returns one sample; two calls draw
    /// independent samples, `polar` reuses the spare variate.
    pub fn normal(&mut self, mu: f32, sigma: f32) -> f32 {
        let u1: f32 = self.uniform01().clamp(f32::EPSILON, 1.0 - f32::EPSILON);
        let u2 = self.uniform01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        mu + sigma * r * theta.sin()
    }
}
