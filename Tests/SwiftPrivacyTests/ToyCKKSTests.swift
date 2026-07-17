import Testing
@testable import SwiftPrivacy  // adjust to your actual target/module name

@Suite("ToyCKKS")
struct ToyCKKSTests {

    @Test("Homomorphic Addition")
    func testHomomorphicAddition() throws {
        let key = ToyCKKS.generateKey()

        let ct1 = ToyCKKS.encrypt(message: 1.25, key: key)
        let ct2 = ToyCKKS.encrypt(message: 3.5, key: key)

        let ctSum = try ToyCKKS.add(ct1, ct2)
        let decrypted = key.decrypt(ctSum)

        // Two independent error terms from errorSampler(), each in [-2, 2] by
        // default, sum to at most 4 in magnitude — a hard bound (every draw is
        // in that range), not a probabilistic one. The old threshold of 0.01
        // was tighter than 4/256 ≈ 0.0156, so it failed on ~24% of runs even
        // for a fully correct implementation.
        let maxNoise = 4.0 / ToyCKKS.defaultScale
        #expect(abs(decrypted - 4.75) < maxNoise + 1e-9)
    }

    @Test("Plaintext Multiplication By A Large Scalar Does Not Trap")
    func testMultiplyLargeScalarDoesNotCrash() {
        let key = ToyCKKS.generateKey()
        let ct = ToyCKKS.encrypt(message: 2.0, key: key)

        // Regression test: previously, Int(round(scalar * ct.scale)) and the
        // subsequent Int64 * Int64 product could exceed Int64's range and trap
        // for large scalars. This should now complete without crashing.
        let hugeScalar = 1e12
        let result = ToyCKKS.multiply(ct, by: hugeScalar)
        #expect(result.a.count == ct.a.count)
    }

    @Test("Scale Mismatch Is Rejected")
    func testScaleMismatchThrows() {
        let key = ToyCKKS.generateKey()
        let ct1 = ToyCKKS.encrypt(message: 1.0, key: key, scale: 256.0)
        let ct2 = ToyCKKS.encrypt(message: 1.0, key: key, scale: 512.0)

        #expect(throws: ToyCKKSError.self) {
            _ = try ToyCKKS.add(ct1, ct2)
        }
    }
}
