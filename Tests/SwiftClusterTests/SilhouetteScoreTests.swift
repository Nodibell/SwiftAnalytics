import XCTest
import SwiftDataFrame
@testable import SwiftCluster

final class SilhouetteScoreTests: XCTestCase {
    
    func testSilhouetteScoreWellSeparatedClusters() throws {
        // Two tight, well-separated Gaussian blobs
        let cluster1: [[Double]] = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.2]
        ]
        let cluster2: [[Double]] = [
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
            [10.0, 10.2]
        ]
        
        let features = cluster1 + cluster2
        let labels = [0, 0, 0, 0, 1, 1, 1, 1]
        
        let score = try SilhouetteScore.compute(features: features, labels: labels)
        XCTAssertGreaterThan(score, 0.85, "Well-separated clusters should have high Silhouette score > 0.85")
    }
    
    func testClusteringMetricsInertiaAndIndices() throws {
        let features: [[Double]] = [
            [0.0, 0.0], [0.5, 0.5],
            [10.0, 10.0], [10.5, 10.5]
        ]
        let labels = [0, 0, 1, 1]
        let centroids = [[0.25, 0.25], [10.25, 10.25]]
        
        let inertia = ClusteringMetrics.inertia(features: features, labels: labels, centroids: centroids)
        XCTAssertGreaterThan(inertia, 0.0)
        XCTAssertLessThan(inertia, 5.0)
        
        let ch = ClusteringMetrics.calinskiHarabaszIndex(features: features, labels: labels)
        XCTAssertGreaterThan(ch, 10.0, "Calinski-Harabasz index should be large for distinct clusters")
        
        let db = ClusteringMetrics.daviesBouldinIndex(features: features, labels: labels)
        XCTAssertGreaterThan(db, 0.0)
        XCTAssertLessThan(db, 0.5, "Davies-Bouldin index should be small for well-separated clusters")
        
        let ari = ClusteringMetrics.adjustedRandIndex(labelsTrue: labels, labelsPred: labels)
        XCTAssertEqual(ari, 1.0, accuracy: 1e-5, "ARI for identical labels should be 1.0")
    }
}
