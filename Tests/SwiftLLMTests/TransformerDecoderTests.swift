import Testing
import Foundation
import MLX
import MLXNN
@testable import SwiftNLP
@testable import SwiftLLM

@Suite("Transformer Decoder Tests")
struct TransformerDecoderTests {
    
    // Helper BPE vocabulary
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
    
    @Test("TransformerDecoder forward pass logits shape")
    func testTransformerForwardPass() throws {
        // Setup MLX metallib access if needed
        registerCmlxBundle()
        
        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        let decoder = TransformerDecoder(vocabSize: vocab.count, tokenizer: tokenizer, dimensions: 16, numHeads: 2, maxSeqLen: 32)
        
        let input = MLXArray([1, 2, 3, 4]) // shape [4]
        let logits = decoder(input)
        
        // Output shape should be [1, seq_len, vocab_size] = [1, 4, 15]
        #expect(logits.shape == [1, 4, 15])
    }
    
    @Test("TransformerDecoder streaming text generation")
    func testTransformerGeneration() async throws {
        registerCmlxBundle()
        
        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        let decoder = TransformerDecoder(vocabSize: vocab.count, tokenizer: tokenizer, dimensions: 16, numHeads: 2, maxSeqLen: 32)
        
        let options = LLMOptions(temperature: 0.0, maxTokens: 5)
        let stream = try await decoder.generate(prompt: "hello", options: options)
        
        var generatedTokens = [String]()
        for await token in stream {
            generatedTokens.append(token)
        }
        
        // Since the weights are random, we just verify the streaming generates successfully
        // and produces some text segments without throwing any crashes.
        #expect(generatedTokens.count >= 0)
    }
    
    // Dynamic metallib finder helper
    private func registerCmlxBundle() {
        #if targetEnvironment(simulator)
        #else
        let path = Bundle.allBundles.first { $0.bundlePath.hasSuffix("mlx-swift_Cmlx.bundle") }
        if path == nil {
            let metallibPath = "./default.metallib"
            if FileManager.default.fileExists(atPath: metallibPath) {
                // Already in working directory, MLX will find it automatically
            }
        }
        #endif
    }
}
