import Foundation

public typealias ErrorSampler = @Sendable () -> Int64

/// A representation of a ciphertext for `ToyCKKS`, storing a fixed-point-encoded real number.
public struct CKKSCiphertext: Sendable, Codable, Equatable {
    public let a: [Int64]
    public let b: Int64
    public let scale: Double
    public let q: Int64

    public init(a: [Int64], b: Int64, scale: Double, q: Int64) {
        self.a = a
        self.b = b
        self.scale = scale
        self.q = q
    }
}

/// A secret key capable of decrypting `ToyCKKS` ciphertexts to real numbers.
public struct CKKSSecretKey: Sendable {
    public let s: [Int64]
    public let q: Int64

    public init(s: [Int64], q: Int64) {
        self.s = s
        self.q = q
    }

    /// Decrypts a ciphertext to a floating-point real number.
    public func decrypt(_ ct: CKKSCiphertext) -> Double {
        let currentQ = ct.q
        var dot: Int64 = 0
        for i in 0..<s.count {
            let ai = ((ct.a[i] % currentQ) + currentQ) % currentQ
            let si = ((s[i] % currentQ) + currentQ) % currentQ
            dot = (dot + ai * si) % currentQ
        }

        let bNorm = ((ct.b % currentQ) + currentQ) % currentQ
        var diff = (bNorm - dot) % currentQ
        if diff < 0 { diff += currentQ }

        // Map modulo space to signed integer space [-q/2, q/2]
        if diff > currentQ / 2 {
            diff -= currentQ
        }

        return Double(diff) / ct.scale
    }
}

/// A toy, single-sample LWE encryption scheme with CKKS-style approximate/rescaling
/// semantics (encrypt / add / plaintext-multiply / rescale over fixed-point reals).
///
/// This is **not** a full Ring-LWE/CKKS implementation. There is no polynomial ring,
/// no NTT, no modulus chain, no relinearization or evaluation keys, and no
/// ciphertext × ciphertext multiplication. It is suitable for teaching and
/// experimentation with the *shape* of CKKS-style homomorphic arithmetic — not for
/// protecting real data, and not a substitute for an audited HE library.
///
/// `d = 8` in particular gives essentially no cryptographic security margin; real
/// (R)LWE parameterizations use ring dimensions in the thousands.
public struct ToyCKKS: Sendable {
    /// 2^31 - 1 (Mersenne prime), chosen to leave headroom under Int64 for the
    /// modular products computed throughout this type.
    public static let q: Int64 = 2_147_483_647
    public static let d = 8
    public static let defaultScale = 256.0

    /// Error-term sampler is now provided per-encryption via a parameter with a default.
    /// Replace the default with a discrete-Gaussian or centered-binomial sampler for
    /// anything beyond teaching/demo purposes.

    private init() {}

    /// Generates a new secret key. Coefficients are drawn from a small range,
    /// which keeps noise growth manageable but is not a strict ternary/binary
    /// secret as used by most modern schemes.
    public static func generateKey() -> CKKSSecretKey {
        let s = (0..<d).map { _ in Int64.random(in: -2...2) }
        return CKKSSecretKey(s: s, q: q)
    }

    /// Encrypts a real number.
    public static func encrypt(message: Double, key: CKKSSecretKey, scale: Double = defaultScale, errorSampler: ErrorSampler = { Int64.random(in: -2...2) }) -> CKKSCiphertext {
        let scaled = Int64(round(message * scale))
        let a = (0..<d).map { _ in Int64.random(in: 0..<q) }
        let e = errorSampler()

        var dot: Int64 = 0
        for i in 0..<d {
            dot = (dot + a[i] * key.s[i]) % q
        }

        let b = (((dot + e + scaled) % q) + q) % q
        return CKKSCiphertext(a: a, b: b, scale: scale, q: q)
    }

    /// Homomorphically adds two ciphertexts: Enc(m1) + Enc(m2) = Enc(m1 + m2).
    public static func add(_ ct1: CKKSCiphertext, _ ct2: CKKSCiphertext) throws -> CKKSCiphertext {
        guard abs(ct1.scale - ct2.scale) < 1e-5 else {
            throw ToyCKKSError.scaleMismatch(ct1.scale, ct2.scale)
        }
        guard ct1.q == ct2.q else {
            throw ToyCKKSError.modulusMismatch(ct1.q, ct2.q)
        }
        let a = zip(ct1.a, ct2.a).map { ((($0 + $1) % ct1.q) + ct1.q) % ct1.q }
        let b = (((ct1.b + ct2.b) % ct1.q) + ct1.q) % ct1.q
        return CKKSCiphertext(a: a, b: b, scale: ct1.scale, q: ct1.q)
    }

    /// Homomorphically adds a plaintext constant: Enc(m) + constant = Enc(m + constant).
    public static func addPlain(_ ct: CKKSCiphertext, _ constant: Double) -> CKKSCiphertext {
        let qd = Double(ct.q)
        // Reduce in Double space first so the Int64 conversion below can never
        // trap, no matter how large `constant` is.
        let scaledReduced = (constant * ct.scale).truncatingRemainder(dividingBy: qd)
        let scaled = Int64(scaledReduced)
        let b = (((ct.b + scaled) % ct.q) + ct.q) % ct.q
        return CKKSCiphertext(a: ct.a, b: b, scale: ct.scale, q: ct.q)
    }

    /// Homomorphically multiplies by a plaintext scalar: Enc(m) * scalar = Enc(m * scalar).
    /// Scales the ciphertext's scale by the scalar's scale.
    public static func multiply(_ ct: CKKSCiphertext, by scalar: Double) -> CKKSCiphertext {
        let qd = Double(ct.q)
        // Reduce in Double space *before* converting to Int64: for a large enough
        // `scalar`, `scalar * ct.scale` can exceed Int64's range, and converting
        // that directly with Int64(round(...)) traps. Reducing mod q in Double
        // space first keeps the value small enough to convert safely, and also
        // keeps the subsequent Int64 * Int64 product safely under Int64.max
        // (bounded by roughly q², about half of Int64.max).
        let sValReduced = (scalar * ct.scale).truncatingRemainder(dividingBy: qd)
        var sVal = Int64(sValReduced)
        sVal = ((sVal % ct.q) + ct.q) % ct.q

        let a = ct.a.map { ((($0 * sVal) % ct.q) + ct.q) % ct.q }
        let b = (((ct.b * sVal) % ct.q) + ct.q) % ct.q
        return CKKSCiphertext(a: a, b: b, scale: ct.scale * ct.scale, q: ct.q)
    }

    /// Rescales a ciphertext to reduce its scale factor and noise.
    ///
    /// NOTE: this divides and re-derives a smaller modulus using floating-point
    /// rounding rather than an exact, modulus-chain-aware switch. It is a
    /// reasonable approximation for teaching purposes but is not the precise
    /// modulus-switching operation a real CKKS implementation performs.
    public static func rescale(_ ct: CKKSCiphertext, targetScale: Double) -> CKKSCiphertext {
        let factor = ct.scale / targetScale
        guard factor > 1.0 else { return ct }

        let newQ = Int64(round(Double(ct.q) / factor))
        let a = ct.a.map { ((Int64(round(Double($0) / factor)) % newQ) + newQ) % newQ }
        let b = ((Int64(round(Double(ct.b) / factor)) % newQ) + newQ) % newQ
        return CKKSCiphertext(a: a, b: b, scale: targetScale, q: newQ)
    }
}

public enum ToyCKKSError: Error, Sendable, Equatable, CustomStringConvertible {
    case scaleMismatch(Double, Double)
    case modulusMismatch(Int64, Int64)

    public var description: String {
        switch self {
        case .scaleMismatch(let a, let b):
            return "Cannot combine ciphertexts with different scales: \(a) vs \(b)"
        case .modulusMismatch(let a, let b):
            return "Cannot combine ciphertexts with different moduli: \(a) vs \(b)"
        }
    }
}

// MARK: - Backward compatibility

/// `CKKS` has been renamed to `ToyCKKS` to make the scope of this type honest:
/// it is a single-sample LWE scheme with CKKS-style semantics, not a full
/// Ring-LWE/CKKS implementation. Update call sites to `ToyCKKS` when convenient;
/// this alias exists so existing code (and any already-tagged release) keeps
/// compiling. Consider this an API change worth a semver-major bump if a
/// version has already shipped under the old name.
@available(*, deprecated, renamed: "ToyCKKS")
public typealias CKKS = ToyCKKS

