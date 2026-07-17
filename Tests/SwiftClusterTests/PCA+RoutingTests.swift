import Testing
import Foundation
@testable import SwiftCluster

@Suite("PCA Routing Tests")
struct PCARoutingTests {
    
    init() {
        if let url = Bundle.main.resourceURL?.appendingPathComponent("mlx-swift_Cmlx.bundle"),
           let bundle = Bundle(url: url) {
            _ = bundle.resourceURL
        }
        for bundle in Bundle.allBundles {
            if let url = bundle.resourceURL?.appendingPathComponent("mlx-swift_Cmlx.bundle"),
               let b = Bundle(url: url) {
                _ = b.resourceURL
            }
        }
    }
    
    @Test("PCA routing CPU vs GPU")
    func testPCARouting() async throws {
        let X: [[Double]] = [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0]
        ]
        
        let pcaCPU = try PCA(nComponents: 1, device: .cpu)
        try await pcaCPU.fit(X)
        
        let pcaGPU = try PCA(nComponents: 1, device: .gpu)
        try await pcaGPU.fit(X)
        
        let devCPU = await pcaCPU.resolvedDevice
        let devGPU = await pcaGPU.resolvedDevice
        
        #expect(devCPU == .cpu)
        #expect(devGPU == .gpu)
        
        let meanCPU = await pcaCPU.mean
        let meanGPU = await pcaGPU.mean
        #expect(meanCPU != nil && meanGPU != nil)
        #expect(abs(meanCPU![0] - meanGPU![0]) < 1e-5)
        #expect(abs(meanCPU![1] - meanGPU![1]) < 1e-5)
        
        let compCPU = await pcaCPU.components
        let compGPU = await pcaGPU.components
        #expect(compCPU != nil && compGPU != nil)
        // Eigenvector sign can flip (arbitrary direction), so check absolute values
        #expect(abs(abs(compCPU![0][0]) - abs(compGPU![0][0])) < 1e-3)
        #expect(abs(abs(compCPU![0][1]) - abs(compGPU![0][1])) < 1e-3)
        
        let projCPU = try await pcaCPU.transform(X)
        let projGPU = try await pcaGPU.transform(X)
        #expect(abs(abs(projCPU[0][0]) - abs(projGPU[0][0])) < 1e-3)
    }
}
