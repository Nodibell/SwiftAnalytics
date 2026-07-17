import Testing
import Foundation
@testable import SwiftForecast

@Suite("GARCH Volatility Model Tests")
struct GARCHTests {
    
    @Test("GARCH(0, 0) constant variance")
    func testGARCH00() async throws {
        let series = [1.0, -1.0, 1.5, -0.5, 0.8, -1.2, 1.1, -0.9, 1.0, -1.0]
        let model = try GARCHModel(p: 0, q: 0)
        
        try await model.fit(series: series)
        let forecast = try await model.forecastVolatility(steps: 3)
        
        #expect(forecast.count == 3)
        let mean = series.reduce(0.0, +) / Double(series.count)
        let variance = series.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(series.count - 1)
        let sampleStdDev = sqrt(variance)
        for val in forecast {
            #expect(abs(val - sampleStdDev) < 0.1)
        }
    }
    
    @Test("GARCH parameter validation")
    func testGARCHValidation() throws {
        #expect(throws: (any Error).self) {
            _ = try GARCHModel(p: -1, q: 1)
        }
        #expect(throws: (any Error).self) {
            _ = try GARCHModel(p: 1, q: -1)
        }
    }
}
