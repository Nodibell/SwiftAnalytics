import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("Pipeline Tests")
struct PipelineTests {
    
    @Test("Pipeline chains Imputer and StandardScaler")
    func testPipelineChaining() throws {
        let imputer = Imputer(strategy: .mean)
        let scaler = StandardScaler()
        let pipeline = Pipeline(steps: [imputer, scaler])
        
        let trainData = [
            [2.0],
            [Double.nan],
            [4.0]
        ]
        
        // Imputation should fill NaN with mean (3.0).
        // Imputed: [2.0], [3.0], [4.0]
        // Mean of imputed is 3.0. Stddev of imputed is sqrt(2/3) ≈ 0.816496580927726
        
        let transformed = try pipeline.fitTransform(trainData)
        
        #expect(abs(transformed[0][0] - (-1.224744871391589)) < 1e-6)
        #expect(abs(transformed[1][0] - 0.0) < 1e-6)
        #expect(abs(transformed[2][0] - 1.224744871391589) < 1e-6)
    }
}
