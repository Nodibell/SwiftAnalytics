import Testing
import Foundation
@testable import SwiftForecast

@Suite("SARIMA Model Tests")
struct SARIMATests {
    
    @Test("SARIMA(0, 0, 0)(0, 0, 0)s on random noise")
    func testSARIMA000() async throws {
        let noise = [1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.0]
        let model = try SARIMAModel(p: 0, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 4)
        
        try await model.fit(series: noise)
        let predictions = try await model.forecast(steps: 3)
        
        #expect(predictions.count == 3)
        let mean = noise.reduce(0.0, +) / Double(noise.count)
        for pred in predictions {
            #expect(abs(pred - mean) < 0.1)
        }
    }
    
    @Test("SARIMA parameter validation")
    func testSARIMAValidation() throws {
        #expect(throws: Error.self) {
            _ = try SARIMAModel(p: -1, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 4)
        }
        #expect(throws: Error.self) {
            _ = try SARIMAModel(p: 0, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 0)
        }
    }
}
