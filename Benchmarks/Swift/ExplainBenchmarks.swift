// ExplainBenchmarks.swift
// Benchmarks for SwiftExplain vs Python SHAP:
//   • KernelSHAP explanation (5 features, 100 coalitions, 20 background samples)

import Foundation
import SwiftExplain

struct ExplainBenchmarks: BenchmarkSuite {
    let module = "SwiftExplain"

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        let M = 5
        let numBackground = 20
        let numCoalitions = 100

        // Background dataset: 20 samples of 5 features
        let background = (0..<numBackground).map { _ in
            (0..<M).map { _ in Double.random(in: -2.0...2.0) }
        }

        // Target instance to explain
        let instance = (0..<M).map { _ in Double.random(in: -2.0...2.0) }

        // Simple linear model: sum of features
        let model: @Sendable ([Double]) -> Double = { x in
            return x.reduce(0.0, +)
        }

        let shapResult = await BenchmarkRunner.run(
            name: "KernelSHAP Explain (5 feats, 100 coalitions)",
            module: module,
            warmup: 2,
            iterations: 5
        ) {
            let explainer = KernelSHAP()
            _ = await explainer.explain(
                model: model,
                instance: instance,
                background: background,
                numCoalitions: numCoalitions
            )
        }
        results.append(shapResult)

        return results
    }
}
