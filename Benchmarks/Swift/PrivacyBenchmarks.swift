// PrivacyBenchmarks.swift
// Benchmarks for SwiftPrivacy vs Python Ring-LWE & PNNS:
//   • Ring-LWE encrypt/decrypt round-trip (vector size 64)
//   • PNNS classification (50 DB vectors, vector size 64)

import Foundation
import SwiftPrivacy

struct PrivacyBenchmarks: BenchmarkSuite {
    let module = "SwiftPrivacy"

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        let vectorSize = 64
        let dbSize = 50

        let key = LWE.generateKey()
        let client = PNNSClient(key: key)

        // 1. Encryption/Decryption Round-Trip
        let plainVector = (0..<vectorSize).map { _ in Int.random(in: 0..<10) }

        let cryptResult = await BenchmarkRunner.run(
            name: "RingLWE Encrypt/Decrypt (vector size=64)",
            module: module,
            warmup: 2,
            iterations: 5
        ) {
            let encrypted = client.encryptQuery(plainVector)
            let decrypted = client.decryptResults(encrypted)
            _ = decrypted.count
        }
        results.append(cryptResult)

        // 2. Private Nearest Neighbor Search
        let db = (0..<dbSize).map { _ in
            (0..<vectorSize).map { _ in Int.random(in: 0..<10) }
        }
        let server = PNNSServer(database: db)

        let pnnsResult = await BenchmarkRunner.run(
            name: "PNNS Classify (50 DB vectors, size=64)",
            module: module,
            warmup: 2,
            iterations: 5
        ) {
            let encryptedQuery = client.encryptQuery(plainVector)
            let secureDots = server.computeSecureDotProducts(encryptedQuery: encryptedQuery)
            let decryptedDots = client.decryptResults(secureDots)
            _ = client.findMostSimilarIndex(decryptedDotProducts: decryptedDots)
        }
        results.append(pnnsResult)

        return results
    }
}
