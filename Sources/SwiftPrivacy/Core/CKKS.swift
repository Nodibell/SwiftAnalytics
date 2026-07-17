import Foundation
#if canImport(SwiftDataFrame)
import SwiftDataFrame
#endif

/// A representation of a CKKS ciphertext storing fixed-point numbers.
public struct CKKSCiphertext: Sendable, Codable, Equatable {
    public let a: [Int]
    public let b: Int
    public let scale: Double
    public let q: Int
    
    public init(a: [Int], b: Int, scale: Double, q: Int) {
        self.a = a
        self.b = b
        self.scale = scale
        self.q = q
    }
}

/// A secret key for the CKKS scheme capable of decrypting real numbers.
public struct CKKSSecretKey: Sendable {
    public let s: [Int]
    public let q: Int
    
    public init(s: [Int], q: Int) {
        self.s = s
        self.q = q
    }
    
    /// Decrypts a CKKS ciphertext to a floating-point real number.
    public func decrypt(_ ct: CKKSCiphertext) -> Double {
        let currentQ = Int64(ct.q)
        var dot: Int64 = 0
        for i in 0..<s.count {
            let ai = (Int64(ct.a[i]) % currentQ + currentQ) % currentQ
            let si = (Int64(s[i]) % currentQ + currentQ) % currentQ
            dot = (dot + ai * si) % currentQ
        }
        var diff = (Int64(ct.b) - dot) % currentQ
        if diff < 0 { diff += currentQ }
        
        // Map modulo space to signed integer space [-q/2, q/2]
        if diff > currentQ / 2 {
            diff -= currentQ
        }
        
        return Double(diff) / ct.scale
    }
}

/// The CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme supporting real-number arithmetic.
public struct CKKS: Sendable {
    public static let q = 2147483647 // 31st Mersenne Prime to prevent modular overflow during multiplication
    public static let d = 8
    public static let defaultScale = 256.0
    
    private init() {}
    
    /// Generates a new CKKS secret key.
    public static func generateKey() -> CKKSSecretKey {
        let s = (0..<d).map { _ in Int.random(in: -2...2) }
        return CKKSSecretKey(s: s, q: q)
    }
    
    /// Encrypts a real number.
    public static func encrypt(message: Double, key: CKKSSecretKey, scale: Double = defaultScale) -> CKKSCiphertext {
        let scaled = Int(round(message * scale))
        let a = (0..<d).map { _ in Int.random(in: 0..<q) }
        let e = Int.random(in: -2...2)
        
        var dot: Int64 = 0
        for i in 0..<d {
            dot = (dot + Int64(a[i]) * Int64(key.s[i])) % Int64(q)
        }
        
        let b = Int((dot + Int64(e) + Int64(scaled)) % Int64(q))
        return CKKSCiphertext(a: a, b: b, scale: scale, q: q)
    }
    
    /// Homomorphically adds two ciphertexts: Enc(m1) + Enc(m2) = Enc(m1 + m2).
    public static func add(_ ct1: CKKSCiphertext, _ ct2: CKKSCiphertext) throws -> CKKSCiphertext {
        guard abs(ct1.scale - ct2.scale) < 1e-5 else {
            #if canImport(SwiftML)
            throw SwiftMLError.invalidInput("Cannot add ciphertexts with different scales: \(ct1.scale) vs \(ct2.scale)")
            #else
            throw NSError(domain: "SwiftPrivacy", code: 1, userInfo: [NSLocalizedDescriptionKey: "Different scales"])
            #endif
        }
        guard ct1.q == ct2.q else {
            #if canImport(SwiftML)
            throw SwiftMLError.invalidInput("Cannot add ciphertexts with different moduli: \(ct1.q) vs \(ct2.q)")
            #else
            throw NSError(domain: "SwiftPrivacy", code: 2, userInfo: [NSLocalizedDescriptionKey: "Different moduli"])
            #endif
        }
        let a = zip(ct1.a, ct2.a).map { ($0 + $1) % ct1.q }
        let b = (ct1.b + ct2.b) % ct1.q
        return CKKSCiphertext(a: a, b: b, scale: ct1.scale, q: ct1.q)
    }
    
    /// Homomorphically adds a plaintext constant: Enc(m) + constant = Enc(m + constant).
    public static func addPlain(_ ct: CKKSCiphertext, _ constant: Double) -> CKKSCiphertext {
        let scaled = Int(round(constant * ct.scale))
        let b = (ct.b + scaled) % ct.q
        return CKKSCiphertext(a: ct.a, b: b, scale: ct.scale, q: ct.q)
    }
    
    /// Homomorphically multiplies by a plaintext scalar: Enc(m) * scalar = Enc(m * scalar).
    /// Scales the ciphertext scale by the scalar scale.
    public static func multiply(_ ct: CKKSCiphertext, by scalar: Double) -> CKKSCiphertext {
        let qi = Int64(ct.q)
        let qd = Double(ct.q)

        let sValReduced = (scalar * ct.scale).truncatingRemainder(dividingBy: qd)
        var sVal = Int64(sValReduced)
        sVal = ((sVal % qi) + qi) % qi

        let a = ct.a.map { Int((Int64($0) * sVal) % qi) }
        let b = Int((Int64(ct.b) * sVal) % qi)
        return CKKSCiphertext(a: a, b: b, scale: ct.scale * ct.scale, q: ct.q)
    }
    
    /// Rescales a ciphertext to reduce its scale factor and noise.
    public static func rescale(_ ct: CKKSCiphertext, targetScale: Double) -> CKKSCiphertext {
        let factor = ct.scale / targetScale
        guard factor > 1.0 else { return ct }
        
        let newQ = Int(round(Double(ct.q) / factor))
        let a = ct.a.map { Int(round(Double($0) / factor)) % newQ }
        let b = Int(round(Double(ct.b) / factor)) % newQ
        return CKKSCiphertext(a: a, b: b, scale: targetScale, q: newQ)
    }
}
