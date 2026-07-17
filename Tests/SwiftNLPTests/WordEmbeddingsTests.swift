import Testing
import Foundation
@testable import SwiftNLP

@Suite("Word Embeddings Tests")
struct WordEmbeddingsTests {
    
    @Test("Word Embeddings cosine similarity and Top-K search")
    func testWordEmbeddings() throws {
        // Define simple dummy embeddings
        let mockEmbeddings: [String: [Double]] = [
            "king":   [0.9, 0.9, 0.1],
            "queen":  [0.8, 0.95, 0.15],
            "man":    [0.85, 0.2, 0.9],
            "woman":  [0.75, 0.25, 0.95],
            "apple":  [0.1, 0.1, 0.2],
            "orange": [0.15, 0.08, 0.25]
        ]
        
        let embeddings = WordEmbeddings(embeddings: mockEmbeddings)
        
        #expect(embeddings.vectorSize == 3)
        
        // Vector retrieval
        let kingVec = embeddings.vector(for: "King") // Test case insensitivity fallback
        #expect(kingVec != nil)
        #expect(kingVec! == [0.9, 0.9, 0.1])
        
        // Cosine similarity checks
        let simKingQueen = try #require(embeddings.cosineSimilarity("king", "queen"))
        let simKingApple = try #require(embeddings.cosineSimilarity("king", "apple"))
        #expect(simKingQueen > 0.9)
        #expect(simKingQueen > simKingApple)
        
        let simAppleOrange = try #require(embeddings.cosineSimilarity("apple", "orange"))
        #expect(simAppleOrange > 0.9)
        
        // Most similar Top-K search
        let similarToKing = try #require(embeddings.mostSimilar(to: "king", topK: 2))
        #expect(similarToKing.count == 2)
        #expect(similarToKing[0].word == "queen")
    }
    
    @Test("Word Embeddings load from text file")
    func testWordEmbeddingsFileLoading() throws {
        // Create a temporary vector file in scratch directory
        let tempDir = FileManager.default.temporaryDirectory
        let filePath = tempDir.appendingPathComponent("test_embeddings.txt").path
        
        let fileContent = """
        4 3
        king 0.9 0.9 0.1
        queen 0.8 0.95 0.15
        apple 0.1 0.1 0.2
        orange 0.15 0.08 0.25
        """
        
        try fileContent.write(toFile: filePath, atomically: true, encoding: .utf8)
        
        let embeddings = try WordEmbeddings(filePath: filePath)
        #expect(embeddings.vectorSize == 3)
        #expect(embeddings.vector(for: "king") != nil)
        #expect(embeddings.vector(for: "apple") != nil)
        
        // Cleanup temp file
        try? FileManager.default.removeItem(atPath: filePath)
    }
}
