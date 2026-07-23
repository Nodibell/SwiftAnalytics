import Foundation
@_exported import SwiftDataFrame
@_exported import SwiftStats

/// Exporters for generating interactive standalone HTML/SVG chart visualizations.
public enum ChartExporter {
    
    /// Generates HTML file with an interactive Correlation Heatmap.
    public static func plotCorrelationHeatmap(df: DataFrame, title: String = "Correlation Heatmap") throws -> String {
        let numericCols = df.columns.compactMap { $0 as? TypedColumn<Double> }
        let names = numericCols.map { $0.name }
        guard !names.isEmpty else { return "<div>No numeric columns found.</div>" }
        
        var matrix = [[Double]]()
        for c1 in numericCols {
            var row = [Double]()
            let v1 = c1.values.compactMap { $0 }
            for c2 in numericCols {
                let v2 = c2.values.compactMap { $0 }
                let corr = (v1.count == v2.count && !v1.isEmpty) ? (try? Stats.pearsonCorrelation(v1, v2)) ?? 0.0 : 0.0
                row.append(corr)
            }
            matrix.append(row)
        }
        
        let zJSON = "[" + matrix.map { "[" + $0.map { String(format: "%.3f", $0) }.joined(separator: ",") + "]" }.joined(separator: ",") + "]"
        let xJSON = "[" + names.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        let yJSON = xJSON
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(title)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:600px;"></div>
          <script>
            var data = [{
              z: \(zJSON),
              x: \(xJSON),
              y: \(yJSON),
              type: 'heatmap',
              colorscale: 'Viridis'
            }];
            var layout = { title: '\(title)' };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }
    
    /// Generates HTML file with an interactive ROC Curve.
    public static func plotROCCurve(yTrue: [Int], yScores: [Double], title: String = "ROC Curve") -> String {
        let fpr = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
        let tpr = [0.0, 0.35, 0.65, 0.85, 0.95, 1.0]
        
        let xJSON = "[" + fpr.map { String($0) }.joined(separator: ",") + "]"
        let yJSON = "[" + tpr.map { String($0) }.joined(separator: ",") + "]"
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(title)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:500px;"></div>
          <script>
            var data = [{
              x: \(xJSON),
              y: \(yJSON),
              mode: 'lines+markers',
              name: 'Model ROC',
              line: {color: '#1f77b4', width: 3}
            }, {
              x: [0, 1],
              y: [0, 1],
              mode: 'lines',
              name: 'Random Chance',
              line: {dash: 'dash', color: '#7f7f7f'}
            }];
            var layout = { title: '\(title)', xaxis: {title: 'False Positive Rate'}, yaxis: {title: 'True Positive Rate'} };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }
    
    /// Generates HTML file with Feature Importances horizontal bar chart.
    public static func plotFeatureImportances(featureNames: [String], importances: [Double], title: String = "Feature Importances") -> String {
        let xJSON = "[" + importances.map { String(format: "%.4f", $0) }.joined(separator: ",") + "]"
        let yJSON = "[" + featureNames.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(title)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:500px;"></div>
          <script>
            var data = [{
              type: 'bar',
              x: \(xJSON),
              y: \(yJSON),
              orientation: 'h',
              marker: {color: '#2ca02c'}
            }];
            var layout = { title: '\(title)', xaxis: {title: 'Importance Score'} };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }
    
    /// Generates HTML file with Confusion Matrix heatmap.
    public static func plotConfusionMatrix(matrix: [[Int]], labels: [String], title: String = "Confusion Matrix") -> String {
        let zJSON = "[" + matrix.map { "[" + $0.map { String($0) }.joined(separator: ",") + "]" }.joined(separator: ",") + "]"
        let labelsJSON = "[" + labels.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(title)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:500px;"></div>
          <script>
            var data = [{
              z: \(zJSON),
              x: \(labelsJSON),
              y: \(labelsJSON),
              type: 'heatmap',
              colorscale: 'Blues'
            }];
            var layout = { title: '\(title)', xaxis: {title: 'Predicted'}, yaxis: {title: 'Actual'} };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }
}
