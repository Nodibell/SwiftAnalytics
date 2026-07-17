import Testing
import Foundation
@testable import SwiftExplain

@Suite("Kernel SHAP Tests")
struct KernelSHAPTests {
    
    @Test("KernelSHAP efficiency and linear coefficient recovery")
    func testKernelSHAP() async throws {
        // Simple linear model: f(x) = 2.5 * x[0] - 4.0 * x[1] + 10.0
        let model: @Sendable ([Double]) -> Double = { x in
            guard x.count == 2 else { return 0.0 }
            return 2.5 * x[0] - 4.0 * x[1] + 10.0
        }
        
        let instance = [4.0, 2.0]
        let background = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 3.0],
            [1.0, 0.0]
        ]
        
        // Calculate background mean manually to verify expectations
        // bgMean[0] = (0+1+2+1)/4 = 1.0
        // bgMean[1] = (0+1+3+0)/4 = 1.0
        let bgMean = [1.0, 1.0]
        
        let explainer = KernelSHAP()
        let shapValues = await explainer.explain(
            model: model,
            instance: instance,
            background: background,
            numCoalitions: 150
        )
        
        #expect(shapValues.count == 2)
        
        let fEmpty = model(bgMean)
        let fFull = model(instance)
        
        let sumShap = shapValues.reduce(0.0, +)
        let expectedDiff = fFull - fEmpty // (2.5*4 - 4*2 + 10) - (2.5*1 - 4*1 + 10) = 12 - 8.5 = 3.5
        
        // 1. Check Efficiency (sum of SHAP values matches prediction difference)
        #expect(abs(sumShap - expectedDiff) < 1e-4)
        
        // 2. Check linear contributions (SHAP value of feature i should be coefficient * (instance[i] - bgMean[i]))
        let expectedShap0 = 2.5 * (instance[0] - bgMean[0]) // 2.5 * (4.0 - 1.0) = 7.5
        let expectedShap1 = -4.0 * (instance[1] - bgMean[1]) // -4.0 * (2.0 - 1.0) = -4.0
        
        #expect(abs(shapValues[0] - expectedShap0) < 1e-2)
        #expect(abs(shapValues[1] - expectedShap1) < 1e-2)
    }
}
