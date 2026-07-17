import Testing
import Foundation
@testable import SwiftNLP

@Suite("BPE Tokenizer Tests")
struct BPETokenizerTests {
    
    @Test("BPE Tokenizer encoding and decoding")
    func testBPETokenizer() throws {
        // Simple vocabulary
        let vocab: [String: Int] = [
            "<unk>": 0,
            "h": 1,
            "e": 2,
            "l": 3,
            "o": 4,
            "w": 5,
            "r": 6,
            "d": 7,
            "l</w>": 8,
            "o</w>": 9,
            "d</w>": 10,
            "he": 11,
            "lo</w>": 12,
            "hello</w>": 13,
            "world</w>": 14
        ]
        
        let merges = [
            "h e",       // he
            "l o</w>",   // lo</w>
            "he l",      // hel
            "hel lo</w>" // hello</w>
        ]
        
        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        
        // Test tokenize
        let tokens = tokenizer.tokenize(text: "hello")
        #expect(tokens.count == 1)
        #expect(tokens[0] == "hello</w>")
        
        // Test encode
        let encoded = tokenizer.encode(text: "hello")
        #expect(encoded == [13])
        
        // Test decode
        let decoded = tokenizer.decode(tokens: [13])
        #expect(decoded == "hello")
        
        // Test round-trip
        let text = "hello world"
        // "world" might not fully merge to "world</w>" because "world</w>" is in vocab
        // but we don't have merges for w-o-r-l-d. So it will split: ["w", "o", "r", "l", "d</w>"]
        let words = tokenizer.tokenize(text: text)
        #expect(words.contains("hello</w>"))
        
        let encodedWords = tokenizer.encode(text: text)
        let decodedWords = tokenizer.decode(tokens: encodedWords)
        #expect(decodedWords == "hello world")
    }
}
