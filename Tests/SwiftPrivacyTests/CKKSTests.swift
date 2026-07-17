import Testing
import Foundation
@testable import SwiftPrivacy

@Suite("CKKS Homomorphic Encryption Scheme Tests")
struct CKKSTests {
    
    @Test("CKKS Encryption and Decryption Round-trip")
    func testCKKSRoundTrip() throws {
        let key = CKKS.generateKey()
        let originalMessage = 3.14159
        
        let ciphertext = CKKS.encrypt(message: originalMessage, key: key)
        let decrypted = key.decrypt(ciphertext)
        
        #expect(abs(decrypted - originalMessage) < 0.01)
    }
    
    @Test("CKKS Homomorphic Addition")
    func testCKKSHomomorphicAddition() throws {
        let key = CKKS.generateKey()
        
        let ct1 = CKKS.encrypt(message: 1.25, key: key)
        let ct2 = CKKS.encrypt(message: 3.5, key: key)
        
        let ctSum = try CKKS.add(ct1, ct2)
        let decrypted = key.decrypt(ctSum)
        
        #expect(abs(decrypted - 4.75) < 0.01)
    }
    
    @Test("CKKS Homomorphic Plaintext Addition")
    func testCKKSHomomorphicPlaintextAddition() throws {
        let key = CKKS.generateKey()
        
        let ct = CKKS.encrypt(message: 2.4, key: key)
        let ctAdded = CKKS.addPlain(ct, 1.6)
        let decrypted = key.decrypt(ctAdded)
        
        #expect(abs(decrypted - 4.0) < 0.01)
    }
    
    @Test("CKKS Homomorphic Multiplication and Rescaling")
    func testCKKSHomomorphicMultiplication() throws {
        let key = CKKS.generateKey()
        
        let ct = CKKS.encrypt(message: 1.5, key: key, scale: 1000.0)
        let ctMul = CKKS.multiply(ct, by: 2.5)
        
        let ctRescaled = CKKS.rescale(ctMul, targetScale: 1000.0)
        
        let decrypted = key.decrypt(ctRescaled)
        #expect(abs(decrypted - 3.75) < 0.05)
    }
}
