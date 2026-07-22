import Foundation
import SwiftStats

/// Feature selector that removes all features whose variance does not meet a threshold.
public final class VarianceThreshold: PreprocessingTransformer, @unchecked Sendable {
    public let threshold: Double
    private var selectedIndices: [Int] = []
    
    public init(threshold: Double = 0.0) {
        self.threshold = threshold
    }
    
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let numSamples = Double(data.count)
        let numFeatures = data[0].count
        
        var selected: [Int] = []
        for f in 0..<numFeatures {
            let col = data.map { $0[f] }
            let mean = col.reduce(0.0, +) / numSamples
            let variance = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
            if variance > threshold {
                selected.append(f)
            }
        }
        self.selectedIndices = selected
    }
    
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
        let indices = selectedIndices.isEmpty ? Array(0..<data[0].count) : selectedIndices
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}

/// Feature selector that retains the K highest scoring features.
///
/// When `targets` are provided to `fit(features:targets:)`, features are ranked
/// by ANOVA F-value (supervised selection). When targets are `nil`, falls back
/// to variance-based ranking (unsupervised).
public final class SelectKBest: PreprocessingTransformer, @unchecked Sendable {
    public let k: Int
    private var selectedIndices: [Int] = []
    
    public init(k: Int) {
        self.k = k
    }
    
    public func fit(_ data: [[Double]]) throws {
        try fit(features: data, targets: nil)
    }
    
    public func fit(features: [[Double]], targets: [Double]?) throws {
        guard !features.isEmpty, !features[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let numFeatures = features[0].count
        let numSamples = Double(features.count)
        let kActual = min(k, numFeatures)
        
        var scores: [(index: Int, score: Double)] = []
        
        if let targets {
            // Supervised: ANOVA F-value per feature (groups = target classes)
            var classIndices: [Double: [Int]] = [:]
            for (i, t) in targets.enumerated() {
                classIndices[t, default: []].append(i)
            }
            
            for f in 0..<numFeatures {
                let groups: [[Double]] = classIndices.values.map { indices in
                    indices.map { features[$0][f] }
                }
                let score: Double
                if let result = try? Stats.oneWayANOVA(groups: groups),
                   !result.fStatistic.isNaN, !result.fStatistic.isInfinite {
                    score = result.fStatistic
                } else {
                    // Fallback to variance when ANOVA cannot be computed
                    // (e.g. only one target class present in the dataset)
                    let col = features.map { $0[f] }
                    let mean = col.reduce(0.0, +) / numSamples
                    score = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
                }
                scores.append((f, score))
            }
        } else {
            // Unsupervised fallback: rank by feature variance
            for f in 0..<numFeatures {
                let col = features.map { $0[f] }
                let mean = col.reduce(0.0, +) / numSamples
                let variance = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
                scores.append((f, variance))
            }
        }
        
        scores.sort { $0.score > $1.score }
        self.selectedIndices = scores.prefix(kActual).map { $0.index }.sorted()
    }
    
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
        let indices = selectedIndices.isEmpty ? Array(0..<min(k, data[0].count)) : selectedIndices
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}
