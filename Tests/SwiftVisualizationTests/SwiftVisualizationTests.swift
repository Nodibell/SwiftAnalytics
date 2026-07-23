import Testing
@testable import SwiftVisualization

@Suite("SwiftVisualization Tests")
struct SwiftVisualizationTests {
    
    @Test("ChartExporter generates HTML strings with Plotly script tags")
    func testChartExporterHTML() throws {
        let colA = TypedColumn<Double>(name: "A", values: [1.0, 2.0, 3.0])
        let colB = TypedColumn<Double>(name: "B", values: [2.0, 4.0, 6.0])
        let df = try DataFrame(columns: [colA, colB])
        
        let heatmapHTML = try ChartExporter.plotCorrelationHeatmap(df: df)
        #expect(heatmapHTML.contains("<!DOCTYPE html>"))
        #expect(heatmapHTML.contains("Plotly.newPlot"))
        
        let rocHTML = ChartExporter.plotROCCurve(yTrue: [1, 0, 1], yScores: [0.9, 0.1, 0.8])
        #expect(rocHTML.contains("ROC Curve"))
        
        let featHTML = ChartExporter.plotFeatureImportances(featureNames: ["F1", "F2"], importances: [0.7, 0.3])
        #expect(featHTML.contains("Feature Importances"))
        
        let cmHTML = ChartExporter.plotConfusionMatrix(matrix: [[10, 2], [1, 15]], labels: ["Class 0", "Class 1"])
        #expect(cmHTML.contains("Confusion Matrix"))
    }
}
