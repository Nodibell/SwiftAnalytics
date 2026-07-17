import Testing
import Foundation
@testable import SwiftPrivacy

@Suite("Private Nearest Neighbor Search Tests")
struct PNNSTests {
    
    @Test("Ring-LWE encryption and decryption round-trip")
    func testLWERoundTrip() throws {
        let key = LWE.generateKey()
        let originalMessage = 42
        
        let ciphertext = LWE.encrypt(message: originalMessage, key: key)
        let decrypted = key.decrypt(ciphertext)
        
        #expect(decrypted == originalMessage)
    }
    
    @Test("Homomorphic Addition and Multiplication")
    func testHomomorphicOperations() throws {
        let key = LWE.generateKey()
        
        // 1. Homomorphic Addition: Enc(12) + Enc(15) = Enc(27)
        let ct1 = LWE.encrypt(message: 12, key: key)
        let ct2 = LWE.encrypt(message: 15, key: key)
        let ctSum = LWE.add(ct1, ct2)
        #expect(key.decrypt(ctSum) == 27)
        
        // 2. Homomorphic Plaintext Multiplication: 6 * Enc(7) = Enc(42)
        let ct3 = LWE.encrypt(message: 7, key: key)
        let ctProduct = LWE.multiply(ct3, by: 6)
        #expect(key.decrypt(ctProduct) == 42)
        
        // 3. Homomorphic Plaintext Addition: Enc(20) + 15 = Enc(35)
        let ct4 = LWE.encrypt(message: 20, key: key)
        let ctPlainAdded = LWE.addPlain(ct4, 15, key: key)
        #expect(key.decrypt(ctPlainAdded) == 35)
    }
    
    @Test("Private Nearest Neighbor Search (PNNS) execution")
    func testPNNSExecution() throws {
        let key = LWE.generateKey()
        let client = PNNSClient(key: key)
        
        // Database of vectors
        let database = [
            [5, 1, 9],    // Row 0
            [10, 8, 2],   // Row 1 (Closest to query)
            [2, 3, 4]     // Row 2
        ]
        let server = PNNSServer(database: database)
        
        // Query vector
        let query = [9, 8, 3]
        
        // 1. Secure Dot Product (Cosine Similarity proxy)
        let encryptedQuery = client.encryptQuery(query)
        let secureDotProducts = server.computeSecureDotProducts(encryptedQuery: encryptedQuery)
        
        let decryptedDots = client.decryptResults(secureDotProducts)
        
        // Let's compute actual dot products manually:
        // Row 0: 5*9 + 1*8 + 9*3 = 45 + 8 + 27 = 80
        // Row 1: 10*9 + 8*8 + 2*3 = 90 + 64 + 6 = 160
        // Row 2: 2*9 + 3*8 + 4*3 = 18 + 24 + 12 = 54
        #expect(decryptedDots[0] == 80)
        #expect(decryptedDots[1] == 160)
        #expect(decryptedDots[2] == 54)
        
        let mostSimilarIndex = try #require(client.findMostSimilarIndex(decryptedDotProducts: decryptedDots))
        #expect(mostSimilarIndex == 1)
        
        // 2. Secure Euclidean Distance: dist^2 = sum(q_j - x_j)^2
        // We need coordinates of query and coordinates of query squared
        let querySquared = query.map { $0 * $0 }
        let encryptedQuerySquared = client.encryptQuery(querySquared)
        
        let secureDistances = server.computeSecureEuclideanDistances(
            encryptedQuery: encryptedQuery,
            encryptedQuerySquared: encryptedQuerySquared,
            clientKey: key
        )
        
        let decryptedDistances = client.decryptResults(secureDistances)
        
        // Let's compute actual squared Euclidean distances:
        // Row 0: (9-5)^2 + (8-1)^2 + (3-9)^2 = 16 + 49 + 36 = 101
        // Row 1: (9-10)^2 + (8-8)^2 + (3-2)^2 = 1 + 0 + 1 = 2 (Minimum distance)
        // Row 2: (9-2)^2 + (8-3)^2 + (3-4)^2 = 49 + 25 + 1 = 75
        #expect(decryptedDistances[0] == 101)
        #expect(decryptedDistances[1] == 2)
        #expect(decryptedDistances[2] == 75)
        
        let nearestNeighborIndex = try #require(client.findNearestNeighborIndex(decryptedDistances: decryptedDistances))
        #expect(nearestNeighborIndex == 1)
    }
}
