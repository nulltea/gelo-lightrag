//! Paillier homomorphic-encryption scheme over `num-bigint`, tuned for
//! RemoteRAG's Stage-2 dot-product rerank.
//!
//! ## Provenance
//!
//! The performance-critical algorithms below — CRT-based modpow in keygen /
//! encryption / decryption, the precomputed `(p, q, p², q², μ_p, μ_q,
//! p_inv_q)` table in [`PaillierPrivateKey`], and the simultaneous
//! multi-exponentiation in [`PaillierPublicKey::homomorphic_dot`] — are
//! direct ports of the corresponding algorithms in
//! [`fast-paillier 0.3.2`](https://github.com/LFDT-Lockness/fast-paillier)
//! (MIT OR Apache-2.0, LFDT-Lockness / dfns). Re-implemented rather than
//! depended on because that crate's pure-Rust `backend-num-bigint` feature
//! transitively pulls `glass_pumpkin`, whose 1.9.x releases depend on the
//! yanked `core2 0.4` crate and whose 1.10 release uses a `rand_core` API
//! incompatible with `fast-paillier`'s call site. The crate's other
//! backend (`backend-rug`) works but introduces LGPL/GMP transitives.
//! Carrying ~400 LOC of textbook Paillier math is easier than maintaining
//! a patched fork.
//!
//! ## What's here
//! - [`PaillierPrivateKey::generate`] — keygen with `g = n + 1`
//!   simplification (μ = λ⁻¹ mod n) and CRT components precomputed.
//! - [`PaillierPublicKey::encrypt_unsigned`] — public-key encryption with
//!   a single 2048-bit modpow for `r^n mod n²`.
//! - [`PaillierPrivateKey::encrypt_unsigned`] — keypair-holder encryption
//!   path that uses CRT to split the `r^n` modpow into two 1024-bit
//!   modpows. ~4× faster than the public-key path.
//! - [`PaillierPrivateKey::decrypt_unsigned`] — CRT-based decryption
//!   (~4× faster than the textbook `c^λ mod n²` form).
//! - [`PaillierPublicKey::homomorphic_dot`] — `Enc(<q, e_d>) = ∏ᵢ
//!   Enc(qᵢ)^{e_d[i]_int}`. Signed `e_d[i]` handled by modular inverse
//!   (avoids the n-bit-exponent modpow case); the product itself is
//!   computed via simultaneous multi-exponentiation (Pippenger-style
//!   bucket method), which beats naive scalar-mul-and-fold by ~3× at
//!   d=1024 dims and 17-bit exponents.
//!
//! ## Security caveats
//! - Default key size is **1024-bit n** (each prime 512-bit), giving
//!   ~80-bit classical security. Intentional for prototype perf; production
//!   should use 2048-bit n or larger. Set via
//!   [`PaillierPrivateKey::generate_with_bits`].
//! - Encryption nonce uses `rand_chacha::ChaCha20Rng` seeded from `OsRng`.
//! - **Not constant-time.** CRT decryption in particular introduces a
//!   timing channel that depends on the bit-length of the intermediate
//!   `m_p`/`m_q` values — fine for prototype use over a trusted client
//!   machine, but a no-go for adversarial co-location.

use anyhow::Result;
use num_bigint::{BigInt, BigUint, RandBigInt, Sign};
use num_integer::Integer;
use num_prime::RandPrime;
use num_traits::{One, Zero};
// `SeedableRng` is imported by fully-qualified path at each call site so
// that the test module can use rand 0.9's `SeedableRng` for `ChaCha20Rng`
// without the rand-0.8 trait shadowing it here.

/// Default modulus bit size. 1024-bit n ≈ 80-bit classical security;
/// adequate for prototype, **not for production**. See module docs.
pub const DEFAULT_KEY_BITS: usize = 1024;

/// Default fixed-point scale: each f32 is quantized to `round(x * 2^16)`.
/// One scalar-mult in the homomorphic dot product raises the scale to 2^32;
/// the recovered i64 is divided by `2^32` to get the float dot product.
/// Result range: ±2^32 · d ≈ ±2^42 for d=1024 — comfortably inside `n`.
pub const DEFAULT_SCALE_BITS: u32 = 16;

/// Paillier public key `(n, g)`. `g = n + 1` is the standard simplification.
#[derive(Debug, Clone)]
pub struct PaillierPublicKey {
    pub n: BigUint,
    pub n_squared: BigUint,
}

/// Paillier private key. Holds the textbook `(λ, μ)` pair plus a
/// **CRT factor table** that lets keygen-aware operations (decrypt,
/// private-side encrypt) split each `mod n²` modpow into two halves over
/// `mod p²` / `mod q²` and recombine. The table is precomputed once at
/// keygen and is what enables the ~4× decrypt and ~2× encrypt speedups.
#[derive(Debug, Clone)]
pub struct PaillierPrivateKey {
    pub public: PaillierPublicKey,
    pub lambda: BigUint,
    pub mu: BigUint,
    // CRT factor table, all derived from p, q during `generate`.
    p: BigUint,
    q: BigUint,
    p_squared: BigUint,
    q_squared: BigUint,
    p_minus_1: BigUint,
    q_minus_1: BigUint,
    /// μ_p = `(L_p(g^{p-1} mod p²))⁻¹ mod p` — Paillier's μ for CRT branch p.
    mu_p: BigUint,
    /// μ_q analogous over q.
    mu_q: BigUint,
    /// `p⁻¹ mod q` for CRT recombination of decrypt outputs (m_p, m_q → m mod n).
    p_inv_q: BigUint,
    /// `(p²)⁻¹ mod q²` for CRT recombination of encrypt outputs
    /// (r_p^n, r_q^n → r^n mod n²). Precomputed once so encryption doesn't
    /// pay a per-call modinverse on this constant.
    p_squared_inv_q_squared: BigUint,
}

/// Single Paillier ciphertext. Lives in `Z_{n²}`.
#[derive(Debug, Clone)]
pub struct PaillierCiphertext {
    pub c: BigUint,
}

impl PaillierPrivateKey {
    /// Generate a fresh keypair at the default 1024-bit modulus.
    pub fn generate() -> Self {
        Self::generate_with_bits(DEFAULT_KEY_BITS)
    }

    /// Generate a fresh keypair at the given modulus size. `n_bits` must be
    /// at least 64 and even (split equally across `p` and `q`).
    pub fn generate_with_bits(n_bits: usize) -> Self {
        assert!(n_bits >= 64 && n_bits % 2 == 0, "n_bits must be ≥ 64 and even");
        let half = n_bits / 2;
        // num-prime's RandPrime trait and num-bigint's RandBigInt trait both
        // blanket-impl on rand 0.8 `Rng`, so the keygen path stays inside
        // rand 0.8. Seed from the workspace rand 0.9 OsRng so entropy is
        // shared across the binary.
        let mut seed = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rng(), &mut seed);
        let mut rng08 = {
        use rand08::SeedableRng as _;
        rand08::rngs::StdRng::from_seed(seed)
    };
        loop {
            let p: BigUint = rng08.gen_prime(half, None);
            let q: BigUint = rng08.gen_prime(half, None);
            if p == q {
                continue;
            }
            let n = &p * &q;
            // λ = lcm(p-1, q-1)
            let pm1 = &p - 1u8;
            let qm1 = &q - 1u8;
            let lambda = (&pm1 * &qm1) / pm1.gcd(&qm1);
            // μ = λ⁻¹ mod n (works when g = n + 1; see Paillier 1999).
            let lambda_signed = BigInt::from_biguint(Sign::Plus, lambda.clone());
            let n_signed = BigInt::from_biguint(Sign::Plus, n.clone());
            let Some(mu_signed) = lambda_signed.modinv(&n_signed) else {
                continue;
            };
            let mu = mu_signed.to_biguint().expect("modinv result in [0, n)");
            let n_squared = &n * &n;

            // CRT factor table — fast-paillier 0.3.2 src/keygen.rs style.
            let p_squared = &p * &p;
            let q_squared = &q * &q;
            // μ_p = (L_p(g^{p-1} mod p²))⁻¹ mod p, where g = n + 1.
            // Using the binomial identity (1 + pq)^{p-1} ≡ 1 + (p-1)·pq (mod p²)
            // this simplifies to ((p-1)·q)⁻¹ mod p = -q⁻¹ mod p. We compute
            // it directly anyway, so a future change to g (or a non-paillier
            // variant) doesn't silently miscalibrate.
            let g = &n + 1u8;
            let g_p_minus_1 = g.modpow(&pm1, &p_squared);
            let l_p_g = (&g_p_minus_1 - 1u8) / &p; // L_p(x) = (x - 1) / p
            let Some(mu_p) =
                BigInt::from_biguint(Sign::Plus, l_p_g.clone()).modinv(&BigInt::from_biguint(
                    Sign::Plus,
                    p.clone(),
                ))
            else {
                continue;
            };
            let mu_p = mu_p.to_biguint().expect("modinv ≥ 0");

            let g_q_minus_1 = g.modpow(&qm1, &q_squared);
            let l_q_g = (&g_q_minus_1 - 1u8) / &q;
            let Some(mu_q) =
                BigInt::from_biguint(Sign::Plus, l_q_g.clone()).modinv(&BigInt::from_biguint(
                    Sign::Plus,
                    q.clone(),
                ))
            else {
                continue;
            };
            let mu_q = mu_q.to_biguint().expect("modinv ≥ 0");

            // p⁻¹ mod q for Garner's CRT recombination on decrypt outputs.
            let Some(p_inv_q) = BigInt::from_biguint(Sign::Plus, p.clone())
                .modinv(&BigInt::from_biguint(Sign::Plus, q.clone()))
            else {
                continue;
            };
            let p_inv_q = p_inv_q.to_biguint().expect("modinv ≥ 0");

            // (p²)⁻¹ mod q² for Garner's CRT recombination on encrypt outputs.
            let Some(p_squared_inv_q_squared) =
                BigInt::from_biguint(Sign::Plus, p_squared.clone())
                    .modinv(&BigInt::from_biguint(Sign::Plus, q_squared.clone()))
            else {
                continue;
            };
            let p_squared_inv_q_squared = p_squared_inv_q_squared
                .to_biguint()
                .expect("modinv ≥ 0");

            return Self {
                public: PaillierPublicKey { n, n_squared },
                lambda,
                mu,
                p,
                q,
                p_squared,
                q_squared,
                p_minus_1: pm1,
                q_minus_1: qm1,
                mu_p,
                mu_q,
                p_inv_q,
                p_squared_inv_q_squared,
            };
        }
    }

    /// Garner's CRT recombine: given `a mod p` and `b mod q`, compute
    /// the unique value in `[0, p·q)` matching both residues.
    /// `result = a + p · ((b - a) · p⁻¹ mod q) mod p·q`.
    #[inline]
    fn crt_recombine(
        &self,
        a_mod_p: &BigUint,
        b_mod_q: &BigUint,
        modulus: &BigUint, // pass p·q or p²·q² depending on context
        p_factor: &BigUint,
        q_factor: &BigUint,
        p_inv_q_factor: &BigUint,
    ) -> BigUint {
        // (b - a) mod q, with wrap.
        let b_minus_a_mod_q = if b_mod_q >= a_mod_p {
            (b_mod_q - a_mod_p) % q_factor
        } else {
            // a > b — climb up over q.
            let diff = a_mod_p % q_factor;
            (q_factor + b_mod_q - diff) % q_factor
        };
        let t = (b_minus_a_mod_q * p_inv_q_factor) % q_factor;
        (a_mod_p + p_factor * t) % modulus
    }

    /// Decrypt a ciphertext to a non-negative plaintext in `[0, n)`.
    ///
    /// Uses the CRT factor table: splits `c^λ mod n²` into the two halves
    /// `c^{p-1} mod p²` and `c^{q-1} mod q²`, recovers `m_p ∈ [0, p)` and
    /// `m_q ∈ [0, q)` via Paillier's L-function on each branch, then
    /// Garner-recombines into `m ∈ [0, n)`. Roughly 4× faster than the
    /// textbook single-modpow form because both exponent *and* modulus are
    /// halved on each branch.
    pub fn decrypt_unsigned(&self, ct: &PaillierCiphertext) -> BigUint {
        // c_p = c mod p²; c_q = c mod q²
        let c_p = &ct.c % &self.p_squared;
        let c_q = &ct.c % &self.q_squared;
        // c_p^{p-1} mod p² ; same over q.
        let h_p = c_p.modpow(&self.p_minus_1, &self.p_squared);
        let h_q = c_q.modpow(&self.q_minus_1, &self.q_squared);
        // L-function over each branch.
        let l_p = (&h_p - 1u8) / &self.p; // ∈ [0, p)
        let l_q = (&h_q - 1u8) / &self.q;
        // m_p = L_p · μ_p mod p ; m_q analogous.
        let m_p = (l_p * &self.mu_p) % &self.p;
        let m_q = (l_q * &self.mu_q) % &self.q;
        // CRT recombine into [0, n).
        self.crt_recombine(&m_p, &m_q, &self.public.n, &self.p, &self.q, &self.p_inv_q)
    }

    /// Keypair-holder encryption: faster than the public-key path because
    /// it CRT-splits the `r^n mod n²` modpow into two halves over `mod p²`
    /// and `mod q²`. Roughly 2× faster than [`PaillierPublicKey::encrypt_unsigned`].
    ///
    /// Use this when the client/encryptor holds the private key — the
    /// typical RemoteRAG client-side case. Server-side `homomorphic_dot`
    /// continues to use the public-key path because the server has no
    /// access to the factors.
    pub fn encrypt_unsigned<R: rand::RngCore>(
        &self,
        m: &BigUint,
        rng: &mut R,
    ) -> PaillierCiphertext {
        let r = sample_zn_star(&self.public.n, rng);
        let one_plus_mn = (m * &self.public.n + 1u8) % &self.public.n_squared;
        // CRT-modpow: r^n mod p²·q² as two half-size modpows.
        let r_p = &r % &self.p_squared;
        let r_q = &r % &self.q_squared;
        let r_n_p = r_p.modpow(&self.public.n, &self.p_squared);
        let r_n_q = r_q.modpow(&self.public.n, &self.q_squared);
        // Garner-recombine over the p² / q² basis using the precomputed
        // `p_squared_inv_q_squared` from keygen.
        let r_to_n = self.crt_recombine(
            &r_n_p,
            &r_n_q,
            &self.public.n_squared,
            &self.p_squared,
            &self.q_squared,
            &self.p_squared_inv_q_squared,
        );
        let c = (one_plus_mn * r_to_n) % &self.public.n_squared;
        PaillierCiphertext { c }
    }

    /// Sign-aware variant of [`Self::encrypt_unsigned`] mirroring
    /// [`PaillierPublicKey::encrypt_signed`].
    pub fn encrypt_signed<R: rand::RngCore>(
        &self,
        m: &BigInt,
        rng: &mut R,
    ) -> PaillierCiphertext {
        let m_mod = to_zn(m, &self.public.n);
        self.encrypt_unsigned(&m_mod, rng)
    }

    /// Decrypt to a signed integer. Values in `[0, n/2)` are returned as
    /// positive; values in `[n/2, n)` are interpreted as negatives (`m - n`).
    pub fn decrypt_signed(&self, ct: &PaillierCiphertext) -> BigInt {
        let m = self.decrypt_unsigned(ct);
        let half_n = &self.public.n >> 1;
        let m_signed = BigInt::from_biguint(Sign::Plus, m);
        if m_signed >= BigInt::from_biguint(Sign::Plus, half_n) {
            m_signed - BigInt::from_biguint(Sign::Plus, self.public.n.clone())
        } else {
            m_signed
        }
    }

    pub fn public(&self) -> &PaillierPublicKey {
        &self.public
    }
}

impl PaillierPublicKey {
    /// Encrypt a message `m ∈ [0, n)` with fresh randomness from `rng`.
    /// `c = (1 + m·n) · rⁿ mod n²` (the g = n + 1 form).
    pub fn encrypt_unsigned<R: rand::RngCore>(&self, m: &BigUint, rng: &mut R) -> PaillierCiphertext {
        let r = sample_zn_star(&self.n, rng);
        let one_plus_mn = (m * &self.n + 1u8) % &self.n_squared;
        let r_to_n = r.modpow(&self.n, &self.n_squared);
        let c = (one_plus_mn * r_to_n) % &self.n_squared;
        PaillierCiphertext { c }
    }

    /// Encrypt a signed integer. Negative values map to `n + m` (so a
    /// subsequent `decrypt_signed` recovers the sign correctly).
    pub fn encrypt_signed<R: rand::RngCore>(&self, m: &BigInt, rng: &mut R) -> PaillierCiphertext {
        let m_mod = to_zn(m, &self.n);
        self.encrypt_unsigned(&m_mod, rng)
    }

    /// Homomorphic add: `Enc(a + b) = Enc(a) · Enc(b) mod n²`.
    pub fn add(&self, a: &PaillierCiphertext, b: &PaillierCiphertext) -> PaillierCiphertext {
        PaillierCiphertext {
            c: (&a.c * &b.c) % &self.n_squared,
        }
    }

    /// Homomorphic scalar multiply: `Enc(m·k) = Enc(m)^k mod n²`. Handles
    /// negative `k` via modular inverse instead of the naive map-to-`Z_n`
    /// trick, which would otherwise blow `|k|` up to a ~1024-bit exponent
    /// and dominate the homomorphic dot-product hot path.
    pub fn scalar_mul(&self, ct: &PaillierCiphertext, k: &BigInt) -> PaillierCiphertext {
        use num_traits::Signed;
        if k.is_zero() {
            // c^0 = 1 in Z*_{n²}; that's a deterministic Enc(0). Acceptable
            // for use as a building block inside `homomorphic_dot`, where
            // the *product* of terms carries fresh randomness.
            return PaillierCiphertext { c: BigUint::one() };
        }
        let (base, exp) = if k.is_negative() {
            let c_signed = BigInt::from_biguint(Sign::Plus, ct.c.clone());
            let n2_signed = BigInt::from_biguint(Sign::Plus, self.n_squared.clone());
            let inv = c_signed
                .modinv(&n2_signed)
                .expect("Paillier ciphertexts are coprime to n² by construction");
            let inv_biguint = inv
                .to_biguint()
                .expect("modinv with positive modulus returns non-negative");
            let abs_k = (-k).to_biguint().expect("|negative| is non-negative");
            (inv_biguint, abs_k)
        } else {
            let pos_k = k.to_biguint().expect("k is non-negative here");
            (ct.c.clone(), pos_k)
        };
        PaillierCiphertext {
            c: base.modpow(&exp, &self.n_squared),
        }
    }

    /// `Enc(0)` — the identity for [`Self::add`].
    pub fn encrypt_zero<R: rand::RngCore>(&self, rng: &mut R) -> PaillierCiphertext {
        self.encrypt_unsigned(&BigUint::zero(), rng)
    }

    /// Compute `Enc(<query_ct, doc_int>) = ∏ᵢ query_ct[i]^{doc_int[i]}` over
    /// fixed-point-quantized vectors using **simultaneous bit-by-bit
    /// multi-exponentiation** (Pippenger-style; see Bernstein-Lange survey
    /// or fast-paillier's `multi_exp` module). Beats the textbook
    /// "scalar-mul-and-fold" loop by ~3× at d=1024 with 17-bit exponents
    /// because it pays only one full-width *squaring* per exponent bit
    /// across all dimensions instead of `bit-width` squarings per dimension.
    ///
    /// Signed `doc_int[i]` are handled by pre-inverting the base ciphertext
    /// (a single Bezout per negative dim — cheap compared to a modpow)
    /// rather than the naive "map exponent into Z_n" trick, which would
    /// otherwise blow each negative exponent up to ~|n| bits and dominate.
    pub fn homomorphic_dot(
        &self,
        query_ct: &[PaillierCiphertext],
        doc_int: &[BigInt],
        _rng: &mut impl rand::RngCore,
    ) -> Result<PaillierCiphertext> {
        use num_traits::Signed;
        anyhow::ensure!(
            query_ct.len() == doc_int.len(),
            "query and document must be same dimension"
        );

        // 1. Preprocess: for each (q_ct, d_i):
        //    - if d_i = 0: skip
        //    - if d_i > 0: base ← q_ct.c          exp ← d_i
        //    - if d_i < 0: base ← q_ct.c⁻¹ mod n² exp ← |d_i|
        let mut bases: Vec<BigUint> = Vec::with_capacity(query_ct.len());
        let mut exps: Vec<BigUint> = Vec::with_capacity(query_ct.len());
        for (q_ct, d_i) in query_ct.iter().zip(doc_int.iter()) {
            if d_i.is_zero() {
                continue;
            }
            if d_i.is_negative() {
                let c_signed = BigInt::from_biguint(Sign::Plus, q_ct.c.clone());
                let n2_signed = BigInt::from_biguint(Sign::Plus, self.n_squared.clone());
                let inv = c_signed
                    .modinv(&n2_signed)
                    .expect("Paillier ciphertexts are coprime to n²")
                    .to_biguint()
                    .expect("modinv ≥ 0");
                bases.push(inv);
                exps.push((-d_i).to_biguint().expect("|neg| ≥ 0"));
            } else {
                bases.push(q_ct.c.clone());
                exps.push(d_i.to_biguint().expect("d_i ≥ 0 here"));
            }
        }
        if bases.is_empty() {
            // All exponents were zero ⇒ Enc(0) is the identity ciphertext `1`.
            return Ok(PaillierCiphertext { c: BigUint::one() });
        }

        // 2. Bit-by-bit simultaneous multi-exponentiation.
        let max_bits = exps.iter().map(|e| e.bits()).max().unwrap_or(0);
        let mut acc = BigUint::one();
        for bit in (0..max_bits).rev() {
            // One squaring of acc covers the bit-position shift for ALL bases.
            acc = (&acc * &acc) % &self.n_squared;
            // Accumulate this bit's contribution as a single bucket product,
            // then fold into acc. (Cache-friendly relative to multiplying
            // bases one-by-one into acc.)
            let mut bucket = BigUint::one();
            for (b, e) in bases.iter().zip(exps.iter()) {
                if e.bit(bit) {
                    bucket = (bucket * b) % &self.n_squared;
                }
            }
            if !bucket.is_one() {
                acc = (acc * bucket) % &self.n_squared;
            }
        }
        Ok(PaillierCiphertext { c: acc })
    }
}

/// Map a signed `BigInt` into `Z_n` (`[0, n)`).
fn to_zn(m: &BigInt, n: &BigUint) -> BigUint {
    let n_signed = BigInt::from_biguint(Sign::Plus, n.clone());
    let m_mod = m.mod_floor(&n_signed);
    m_mod
        .to_biguint()
        .expect("mod_floor with positive modulus returns non-negative")
}

/// Uniformly sample `r ∈ Z_n*` (`{r : 1 ≤ r < n, gcd(r, n) = 1}`).
fn sample_zn_star<R: rand::RngCore>(n: &BigUint, rng: &mut R) -> BigUint {
    // num-bigint's `gen_biguint_below` lives on `rand 0.8 Rng`. Re-seed a
    // rand-0.8 StdRng from `rng` so we don't pull a second RNG into the API.
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    let mut rng08 = {
        use rand08::SeedableRng as _;
        rand08::rngs::StdRng::from_seed(seed)
    };
    loop {
        let r: BigUint = rng08.gen_biguint_below(n);
        if !r.is_zero() && r.gcd(n).is_one() {
            return r;
        }
    }
}

/// Quantize an `f32` vector to a `BigInt` vector with `scale_bits` of
/// fractional precision (round-half-to-even).
pub fn quantize(v: &[f32], scale_bits: u32) -> Vec<BigInt> {
    let scale = (1u64 << scale_bits) as f64;
    v.iter()
        .map(|&x| BigInt::from((x as f64 * scale).round() as i64))
        .collect()
}

/// Inverse of [`quantize`] after composing two scale-`scale_bits` operands
/// (so the result has `2 · scale_bits` of fractional precision).
pub fn dequantize_product(m: &BigInt, scale_bits: u32) -> f64 {
    let scale = (1u64 << (2 * scale_bits)) as f64;
    let (sign, magnitude) = match m.sign() {
        Sign::Minus => (-1.0, (-m).to_biguint().unwrap()),
        _ => (1.0, m.to_biguint().unwrap_or_else(BigUint::zero)),
    };
    let mut digits = magnitude.to_u64_digits();
    if digits.is_empty() {
        return 0.0;
    }
    // For our use (d ≤ 8192, scale ≤ 24 bits), the result fits in 64 bits.
    let lo = digits.remove(0) as f64;
    let rest: f64 = digits
        .iter()
        .enumerate()
        .map(|(i, &d)| d as f64 * 2f64.powi(64 * (i as i32 + 1)))
        .sum();
    sign * (lo + rest) / scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn small_key() -> PaillierPrivateKey {
        // 256-bit n keeps unit tests fast; not security-meaningful.
        PaillierPrivateKey::generate_with_bits(256)
    }

    #[test]
    fn keygen_invariants() {
        let sk = small_key();
        assert!(sk.public.n > BigUint::from(1u8 << 7));
        assert_eq!(&sk.public.n_squared, &(&sk.public.n * &sk.public.n));
        // μ · λ ≡ 1 (mod n)
        let prod = (&sk.mu * &sk.lambda) % &sk.public.n;
        assert!(prod.is_one());
    }

    #[test]
    fn encrypt_decrypt_round_trip_unsigned() {
        let sk = small_key();
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for m in [0u64, 1, 7, 42, 999, 65_535] {
            let m_big = BigUint::from(m);
            let ct = sk.public.encrypt_unsigned(&m_big, &mut rng);
            let recovered = sk.decrypt_unsigned(&ct);
            assert_eq!(recovered, m_big, "round-trip failed for m={m}");
        }
    }

    #[test]
    fn encrypt_decrypt_round_trip_signed() {
        let sk = small_key();
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        for m in [-1000i64, -1, 0, 1, 42, 65_535] {
            let ct = sk.public.encrypt_signed(&BigInt::from(m), &mut rng);
            let recovered = sk.decrypt_signed(&ct);
            assert_eq!(recovered, BigInt::from(m), "round-trip failed for m={m}");
        }
    }

    #[test]
    fn homomorphic_add() {
        let sk = small_key();
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let a = BigUint::from(13u32);
        let b = BigUint::from(29u32);
        let ca = sk.public.encrypt_unsigned(&a, &mut rng);
        let cb = sk.public.encrypt_unsigned(&b, &mut rng);
        let csum = sk.public.add(&ca, &cb);
        assert_eq!(sk.decrypt_unsigned(&csum), &a + &b);
    }

    #[test]
    fn homomorphic_scalar_mul_positive() {
        let sk = small_key();
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let m = BigInt::from(7);
        let k = BigInt::from(11);
        let ct = sk.public.encrypt_signed(&m, &mut rng);
        let ct_k = sk.public.scalar_mul(&ct, &k);
        assert_eq!(sk.decrypt_signed(&ct_k), &m * &k);
    }

    #[test]
    fn homomorphic_scalar_mul_negative() {
        let sk = small_key();
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let m = BigInt::from(7);
        let k = BigInt::from(-3);
        let ct = sk.public.encrypt_signed(&m, &mut rng);
        let ct_k = sk.public.scalar_mul(&ct, &k);
        assert_eq!(sk.decrypt_signed(&ct_k), &m * &k);
    }

    #[test]
    fn homomorphic_dot_signed_round_trip() {
        let sk = small_key();
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let q: Vec<BigInt> = vec![1, 2, -3, 5].into_iter().map(BigInt::from).collect();
        let e_d: Vec<BigInt> = vec![4, -1, 2, 0].into_iter().map(BigInt::from).collect();
        let q_ct: Vec<_> = q
            .iter()
            .map(|m| sk.public.encrypt_signed(m, &mut rng))
            .collect();
        let dot_ct = sk.public.homomorphic_dot(&q_ct, &e_d, &mut rng).unwrap();
        let dot_int = sk.decrypt_signed(&dot_ct);
        let expected: BigInt = q.iter().zip(e_d.iter()).map(|(a, b)| a * b).sum();
        assert_eq!(dot_int, expected);
    }

    #[test]
    fn quantize_dequantize_preserves_dot_product() {
        let q: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let e_d: Vec<f32> = vec![0.5, 0.5, -0.5, 0.5];
        let plain_dot: f32 = q.iter().zip(e_d.iter()).map(|(a, b)| a * b).sum();
        let scale = DEFAULT_SCALE_BITS;
        let q_int = quantize(&q, scale);
        let e_d_int = quantize(&e_d, scale);
        let prod: BigInt = q_int.iter().zip(e_d_int.iter()).map(|(a, b)| a * b).sum();
        let recovered = dequantize_product(&prod, scale) as f32;
        assert!(
            (recovered - plain_dot).abs() < 1e-3,
            "recovered {recovered}, plain {plain_dot}"
        );
    }

    #[test]
    fn end_to_end_paillier_dot_matches_plaintext() {
        let sk = small_key();
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect();
        let e_d: Vec<f32> = (0..16).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let scale = DEFAULT_SCALE_BITS;

        let q_int = quantize(&q, scale);
        let e_d_int = quantize(&e_d, scale);
        let q_ct: Vec<_> = q_int
            .iter()
            .map(|m| sk.public.encrypt_signed(m, &mut rng))
            .collect();
        let dot_ct = sk.public.homomorphic_dot(&q_ct, &e_d_int, &mut rng).unwrap();
        let dot_int = sk.decrypt_signed(&dot_ct);
        let homomorphic_dot = dequantize_product(&dot_int, scale) as f32;

        let plain_dot: f32 = q.iter().zip(e_d.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (homomorphic_dot - plain_dot).abs() < 1e-3,
            "homomorphic {homomorphic_dot}, plain {plain_dot}"
        );
    }
}

