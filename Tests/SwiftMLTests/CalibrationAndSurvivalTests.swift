import XCTest
import SwiftStats
@testable import SwiftML

final class CalibrationAndSurvivalTests: XCTestCase {

    func testPlattScalingCalibration() {
        let scores: [Double] = [-2.0, -1.0, 0.0, 1.0, 2.0]
        let targets: [Int] = [0, 0, 0, 1, 1]

        let platt = PlattScaling()
        platt.fit(scores: scores, targets: targets)

        XCTAssertTrue(platt.isFitted)
        let probLow = platt.predictProbability(score: -2.0)
        let probHigh = platt.predictProbability(score: 2.0)

        XCTAssertLessThan(probLow, probHigh, "Lower logit score should yield lower calibrated probability")
        XCTAssertGreaterThan(probHigh, 0.5)
    }

    func testIsotonicRegressionCalibration() {
        let scores: [Double] = [0.1, 0.2, 0.3, 0.8, 0.9]
        let targets: [Int] = [0, 0, 1, 1, 1]

        let iso = IsotonicRegression()
        iso.fit(scores: scores, targets: targets)

        XCTAssertTrue(iso.isFitted)
        let probLow = iso.predictProbability(score: 0.15)
        let probHigh = iso.predictProbability(score: 0.85)

        XCTAssertLessThanOrEqual(probLow, probHigh)
    }

    func testKaplanMeierSurvivalEstimation() {
        let durations: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let events: [Int] = [1, 1, 0, 1, 1]

        let km = KaplanMeier(durations: durations, events: events)
        XCTAssertEqual(km.timeline.count, 5)

        let s1 = km.predictSurvivalProbability(at: 0.5)
        let s5 = km.predictSurvivalProbability(at: 5.5)

        XCTAssertEqual(s1, 1.0, accuracy: 1e-5)
        XCTAssertLessThan(s5, s1)
    }

    func testCoxProportionalHazardsModel() {
        let features: [[Double]] = [
            [0.1], [0.2], [1.5], [1.8]
        ]
        let durations: [Double] = [10.0, 9.0, 2.0, 1.0]
        let events: [Int] = [1, 1, 1, 1]

        let cox = CoxProportionalHazards()
        cox.fit(features: features, durations: durations, events: events)

        XCTAssertTrue(cox.isFitted)
        let hazardLow = cox.predictPartialHazard(features: [0.1])
        let hazardHigh = cox.predictPartialHazard(features: [1.8])

        XCTAssertGreaterThan(hazardHigh, hazardLow, "Higher feature risk factor should produce higher hazard ratio")
    }
}
