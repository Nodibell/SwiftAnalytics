import Testing
import Foundation
@testable import SwiftCluster

@Suite("DBSCAN Tests")
struct DBSCANTests {
    
    @Test("DBSCAN basic clustering")
    func testDBSCANBasic() async throws {
        // Two separate dense clusters and one noise point
        let features = [
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [1.1, 1.1],
            [1.0, 0.9],
            
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.0, 9.9],
            
            [5.0, 5.0] // Noise point
        ]
        
        let dbscan = DBSCAN(eps: 0.5, minSamples: 4)
        try await dbscan.fit(features: features)
        
        let labels = await dbscan.labels
        #expect(labels != nil)
        #expect(labels!.count == 11)
        
        // Cluster 0
        #expect(labels![0] == 0)
        #expect(labels![1] == 0)
        #expect(labels![2] == 0)
        #expect(labels![3] == 0)
        #expect(labels![4] == 0)
        
        // Cluster 1
        #expect(labels![5] == 1)
        #expect(labels![6] == 1)
        #expect(labels![7] == 1)
        #expect(labels![8] == 1)
        #expect(labels![9] == 1)
        
        // Noise point
        #expect(labels![10] == -1)
    }
}
