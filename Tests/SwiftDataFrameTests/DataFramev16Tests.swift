import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("DataFrame v1.6 Core & Matrix Tests")
struct DataFramev16Tests {
    @Test("toFeatureMatrix and toTargetVector extract numeric & boolean columns")
    func testToFeatureMatrix() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Double>(name: "age", values: [25.0, 30.0, 35.0]),
            TypedColumn<Int64>(name: "pclass", values: [1, 2, 3]),
            TypedColumn<Bool>(name: "survived", values: [true, false, true])
        ])

        let matrix = try df.toFeatureMatrix(["age", "pclass", "survived"])
        #expect(matrix.count == 3)
        #expect(matrix[0] == [25.0, 1.0, 1.0])
        #expect(matrix[1] == [30.0, 2.0, 0.0])
        #expect(matrix[2] == [35.0, 3.0, 1.0])

        let target = try df.toTargetVector("survived")
        #expect(target == [1.0, 0.0, 1.0])
    }

    @Test("CSVReadOptions columnTypeOverrides forces column type")
    func testCSVTypeOverrides() throws {
        let csv = """
        Survived,Pclass
        1,3
        0,1
        """
        var options = CSVReadOptions()
        options.columnTypeOverrides["Survived"] = .int64
        let df = try CSVReader.parse(csv, options: options)

        #expect(df[column: "Survived", as: Int64.self] != nil)
        #expect(df[column: "Survived", as: Bool.self] == nil)
    }
}
