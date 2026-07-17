import Foundation

/// A representation of an LWE (Learning With Errors) ciphertext.
public struct LWECiphertext: Sendable, Codable, Equatable {
    public let a: [Int]
    public let b: Int
    
    public init(a: [Int], b: Int) {
        self.a = a
        self.b = b
    }
}

/// A secret key for LWE homomorphic encryption, capable of decrypting ciphertexts.
public struct LWESecretKey: Sendable {
    public let s: [Int]
    public let q: Int
    public let t: Int
    public let delta: Int
    
    public init(s: [Int], q: Int, t: Int) {
        self.s = s
        self.q = q
        self.t = t
        self.delta = q / t
    }
    
    /// Decrypts the LWE ciphertext back to a plaintext integer in [0, t-1].
    public func decrypt(_ ct: LWECiphertext) -> Int {
        var dot = 0
        for i in 0..<s.count {
            dot = (dot + ct.a[i] * s[i]) % q
        }
        var diff = (ct.b - dot) % q
        if diff < 0 { diff += q }
        
        let m = Int(round(Double(diff) / Double(delta))) % t
        return m >= 0 ? m : m + t
    }
}

/// A self-contained symmetric LWE (Learning With Errors) homomorphic encryption scheme.
public struct LWE: Sendable {
    public static let q = 1048583
    public static let t = 1000
    public static let d = 8
    
    private init() {}
    
    /// Generates a new secret key.
    public static func generateKey() -> LWESecretKey {
        let s = (0..<d).map { _ in Int.random(in: 0..<t) }
        return LWESecretKey(s: s, q: q, t: t)
    }
    
    /// Encrypts an integer message.
    public static func encrypt(message: Int, key: LWESecretKey) -> LWECiphertext {
        let a = (0..<d).map { _ in Int.random(in: 0..<q) }
        let e = Int.random(in: -3...3)
        
        var dot = 0
        for i in 0..<d {
            dot = (dot + a[i] * key.s[i]) % q
        }
        
        let scaledMessage = (message * key.delta) % q
        let b = (dot + e + scaledMessage) % q
        
        return LWECiphertext(a: a, b: b)
    }
    
    /// Homomorphically adds two ciphertexts: Enc(m1) + Enc(m2) = Enc(m1 + m2).
    public static func add(_ ct1: LWECiphertext, _ ct2: LWECiphertext) -> LWECiphertext {
        let a = zip(ct1.a, ct2.a).map { ($0 + $1) % q }
        let b = (ct1.b + ct2.b) % q
        return LWECiphertext(a: a, b: b)
    }
    
    /// Homomorphically multiplies a ciphertext by a plaintext scalar: c * Enc(m) = Enc(c * m).
    public static func multiply(_ ct: LWECiphertext, by scalar: Int) -> LWECiphertext {
        // Handle negative scalars by computing modulo q
        let sMod = (scalar % q + q) % q
        let a = ct.a.map { ($0 * sMod) % q }
        let b = (ct.b * sMod) % q
        return LWECiphertext(a: a, b: b)
    }
    
    /// Homomorphically adds a plaintext constant to a ciphertext: Enc(m) + constant = Enc(m + constant).
    public static func addPlain(_ ct: LWECiphertext, _ constant: Int, key: LWESecretKey) -> LWECiphertext {
        let scaledConstant = (constant * key.delta) % q
        let b = (ct.b + scaledConstant) % q
        return LWECiphertext(a: ct.a, b: b)
    }
}
