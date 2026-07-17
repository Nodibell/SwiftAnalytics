// LLMBenchmarks.swift
// Benchmarks for SwiftLLM vs Python PyTorch:
//   • Forward Pass (seqLen=64, vocabSize=1000, dim=64, heads=4)
//   • Token Generation (10 tokens, temp=0.7)

import Foundation
import MLX
import MLXNN
import SwiftNLP
import SwiftLLM

struct LLMBenchmarks: BenchmarkSuite {
    let module = "SwiftLLM"

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        let vocabSize = 1000
        let dimensions = 64
        let numHeads = 4
        let maxSeqLen = 128

        var vocab: [String: Int] = ["<unk>": 0, "hello": 1, "world": 2]
        for i in 3..<vocabSize {
            vocab["tok\(i)"] = i
        }
        let merges: [String] = []
        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)

        let decoder = TransformerDecoder(
            vocabSize: vocabSize,
            tokenizer: tokenizer,
            dimensions: dimensions,
            numHeads: numHeads,
            maxSeqLen: maxSeqLen
        )

        // 1. Forward Pass
        let inputArray = (0..<64).map { $0 % vocabSize }
        let inputMLX = MLXArray(inputArray)

        let forwardResult = await BenchmarkRunner.run(
            name: "LLM Forward Pass (seqLen=64)",
            module: module,
            warmup: 2,
            iterations: 5
        ) {
            let logits = decoder(inputMLX)
            _ = logits.shape
            MLX.eval(logits)
        }
        results.append(forwardResult)

        // 2. Token Generation
        let options = LLMOptions(temperature: 0.7, topK: 50, maxTokens: 10)
        let genResult = await BenchmarkRunner.run(
            name: "LLM Generate (10 tokens)",
            module: module,
            warmup: 1,
            iterations: 3
        ) {
            let stream = try await decoder.generate(prompt: "hello", options: options)
            var count = 0
            for await _ in stream {
                count += 1
            }
        }
        results.append(genResult)

        return results
    }
}
