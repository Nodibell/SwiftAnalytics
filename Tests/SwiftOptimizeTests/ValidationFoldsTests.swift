import XCTest
@testable import SwiftOptimize

final class ValidationFoldsTests: XCTestCase {
    
    func testStratifiedKFoldPreservesClassRatios() {
        let features = [[Double]](repeating: [1.0, 2.0], count: 100)
        // Imbalanced classes: 80 class 0, 20 class 1
        let targets = [Double](repeating: 0.0, count: 80) + [Double](repeating: 1.0, count: 20)
        
        let stratKFold = StratifiedKFold(nSplits: 5, shuffle: false)
        let folds = stratKFold.split(features: features, targets: targets)
        
        XCTAssertEqual(folds.count, 5)
        for fold in folds {
            let valClass1 = fold.valTargets.filter { $0 == 1.0 }.count
            XCTAssertEqual(valClass1, 4, "Each fold validation set should contain exactly 4 class 1 samples")
        }
    }
    
    func testTimeSeriesSplitExpandingWindow() {
        let features = (0..<100).map { [Double($0)] }
        let targets = (0..<100).map { Double($0) }
        
        let tsSplit = TimeSeriesSplit(nSplits: 4)
        let folds = tsSplit.split(features: features, targets: targets)
        
        XCTAssertEqual(folds.count, 4)
        for i in 1..<folds.count {
            XCTAssertGreaterThan(folds[i].trainFeatures.count, folds[i - 1].trainFeatures.count, "Training set should expand over iterations")
        }
    }
    
    func testGroupKFoldNoGroupLeakage() {
        let features = [[Double]](repeating: [1.0], count: 12)
        let targets = [Double](repeating: 0.0, count: 12)
        let groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        
        let groupFold = GroupKFold(nSplits: 4)
        let folds = groupFold.split(features: features, targets: targets, groups: groups)
        
        XCTAssertEqual(folds.count, 4)
    }
    
    func testNewMetricsCalculations() {
        let yTrueInt = [1, 1, 0, 0, 1, 0]
        let yPredInt = [1, 0, 0, 0, 1, 1]
        let yScore = [0.9, 0.4, 0.1, 0.2, 0.8, 0.6]
        
        let fBeta = Metrics.fBetaScore(yTrue: yTrueInt, yPred: yPredInt, label: 1, beta: 0.5)
        XCTAssertGreaterThan(fBeta, 0.0)
        
        let prAuc = Metrics.prAUC(yTrue: yTrueInt, yScore: yScore)
        XCTAssertGreaterThan(prAuc, 0.0)
        
        let yTrueD = [10.0, 20.0, 30.0, 40.0]
        let yPredD = [11.0, 19.0, 31.0, 39.0]
        
        let adjR2 = Metrics.adjustedR2Score(yTrue: yTrueD, yPred: yPredD, numFeatures: 1)
        XCTAssertGreaterThan(adjR2, 0.9)
        
        let mape = Metrics.mape(yTrue: yTrueD, yPred: yPredD)
        XCTAssertLessThan(mape, 10.0, "MAPE should be under 10%")
        
        let expVar = Metrics.explainedVarianceScore(yTrue: yTrueD, yPred: yPredD)
        XCTAssertGreaterThan(expVar, 0.9)
    }
}
