import Foundation
import SwiftDataFrame

extension DataFrame {
    private func extractFeatures(columns names: [String], allowNaN: Bool = false) throws -> [[Double]] {
        let nRows = shape.rows
        guard nRows > 0 else { return [] }
        
        var features = [[Double]](repeating: [Double](repeating: 0.0, count: names.count), count: nRows)
        for (colIdx, colName) in names.enumerated() {
            guard let col = self[column: colName, as: Double.self] else {
                throw DataFrameError.columnNotFound(colName)
            }
            for rowIdx in 0..<nRows {
                features[rowIdx][colIdx] = col[rowIdx] ?? (allowNaN ? Double.nan : 0.0)
            }
        }
        return features
    }

    /// Fits a StandardScaler on the specified columns.
    public func fitStandardScaler(columns names: [String]) throws -> StandardScaler {
        let features = try extractFeatures(columns: names)
        let scaler = StandardScaler()
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted StandardScaler, returning a new DataFrame.
    public func standardScale(columns names: [String], scaler: StandardScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using StandardScaler, returning the scaled DataFrame and scaler.
    public func standardScale(columns names: [String]) throws -> (scaled: DataFrame, scaler: StandardScaler) {
        let scaler = try fitStandardScaler(columns: names)
        let scaledDf = try standardScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }

    /// Fits a MinMaxScaler on the specified columns.
    public func fitMinMaxScaler(columns names: [String]) throws -> MinMaxScaler {
        let features = try extractFeatures(columns: names)
        let scaler = MinMaxScaler()
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted MinMaxScaler, returning a new DataFrame.
    public func minMaxScale(columns names: [String], scaler: MinMaxScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using MinMaxScaler, returning the scaled DataFrame and scaler.
    public func minMaxScale(columns names: [String]) throws -> (scaled: DataFrame, scaler: MinMaxScaler) {
        let scaler = try fitMinMaxScaler(columns: names)
        let scaledDf = try minMaxScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }

    /// Fits a LabelEncoder and encodes a category column into integers.
    public func labelEncode(column name: String) throws -> (encoded: DataFrame, encoder: LabelEncoder) {
        guard let col = self[column: name, as: String.self] else {
            throw DataFrameError.columnNotFound(name)
        }
        let rawValues = col.values.map { $0 ?? "" }
        let encoder = LabelEncoder()
        encoder.fit(rawValues)
        let labels = try encoder.transform(rawValues)
        
        let intValues: [Int64?] = labels.map { Int64($0) }
        let newCol = TypedColumn<Int64>(name: name, values: intValues)
        let encodedDf = try withColumn(name, column: newCol)
        return (encodedDf, encoder)
    }

    // MARK: - Imputer
    
    /// Fits an Imputer on the specified columns.
    public func fitImputer(columns names: [String], strategy: Imputer.Strategy = .mean) throws -> Imputer {
        let features = try extractFeatures(columns: names, allowNaN: true)
        let imputer = Imputer(strategy: strategy)
        try imputer.fit(features)
        return imputer
    }
    
    /// Imputes the specified columns using a fitted Imputer, returning a new DataFrame.
    public func impute(columns names: [String], imputer: Imputer) throws -> DataFrame {
        let features = try extractFeatures(columns: names, allowNaN: true)
        let imputed = try imputer.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in imputed[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and imputes the specified columns, returning the imputed DataFrame and the imputer.
    public func impute(columns names: [String], strategy: Imputer.Strategy = .mean) throws -> (imputed: DataFrame, imputer: Imputer) {
        let imputer = try fitImputer(columns: names, strategy: strategy)
        let imputedDf = try impute(columns: names, imputer: imputer)
        return (imputedDf, imputer)
    }
    
    // MARK: - Normalizer
    
    /// Normalizes the specified columns, returning the normalized DataFrame and normalizer.
    public func normalize(columns names: [String], norm: Normalizer.NormType = .l2) throws -> (normalized: DataFrame, normalizer: Normalizer) {
        let features = try extractFeatures(columns: names)
        let normalizer = Normalizer(norm: norm)
        try normalizer.fit(features)
        let normalized = try normalizer.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in normalized[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return (df, normalizer)
    }
    
    // MARK: - RobustScaler
    
    /// Fits a RobustScaler on the specified columns.
    public func fitRobustScaler(columns names: [String], withCentering: Bool = true, withScaling: Bool = true, quantileRange: (Double, Double) = (25.0, 75.0)) throws -> RobustScaler {
        let features = try extractFeatures(columns: names)
        let scaler = RobustScaler(withCentering: withCentering, withScaling: withScaling, quantileRange: quantileRange)
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted RobustScaler, returning a new DataFrame.
    public func robustScale(columns names: [String], scaler: RobustScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using RobustScaler, returning the scaled DataFrame and scaler.
    public func robustScale(columns names: [String], withCentering: Bool = true, withScaling: Bool = true, quantileRange: (Double, Double) = (25.0, 75.0)) throws -> (scaled: DataFrame, scaler: RobustScaler) {
        let scaler = try fitRobustScaler(columns: names, withCentering: withCentering, withScaling: withScaling, quantileRange: quantileRange)
        let scaledDf = try robustScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }
    
    // MARK: - PowerTransformer
    
    /// Fits a PowerTransformer on the specified columns.
    public func fitPowerTransformer(columns names: [String], method: PowerTransformer.Method = .yeoJohnson, standardize: Bool = true) throws -> PowerTransformer {
        let features = try extractFeatures(columns: names)
        let transformer = PowerTransformer(method: method, standardize: standardize)
        try transformer.fit(features)
        return transformer
    }
    
    /// Transforms the specified columns using a fitted PowerTransformer, returning a new DataFrame.
    public func powerTransform(columns names: [String], transformer: PowerTransformer) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let transformed = try transformer.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in transformed[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and transforms the specified columns using PowerTransformer, returning the transformed DataFrame and transformer.
    public func powerTransform(columns names: [String], method: PowerTransformer.Method = .yeoJohnson, standardize: Bool = true) throws -> (transformed: DataFrame, transformer: PowerTransformer) {
        let transformer = try fitPowerTransformer(columns: names, method: method, standardize: standardize)
        let transformedDf = try powerTransform(columns: names, transformer: transformer)
        return (transformedDf, transformer)
    }
    
    // MARK: - KBinsDiscretizer
    
    /// Fits a KBinsDiscretizer on the specified columns.
    public func fitKBinsDiscretizer(columns names: [String], nBins: Int = 5, strategy: KBinsDiscretizer.Strategy = .uniform, encode: KBinsDiscretizer.Encode = .ordinal) throws -> KBinsDiscretizer {
        let features = try extractFeatures(columns: names)
        let discretizer = KBinsDiscretizer(nBins: nBins, strategy: strategy, encode: encode)
        try discretizer.fit(features)
        return discretizer
    }
    
    /// Discretizes the specified columns using a fitted KBinsDiscretizer, returning a new DataFrame.
    public func kBinsDiscretize(columns names: [String], discretizer: KBinsDiscretizer) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let binned = try discretizer.transform(features)
        
        var df = self
        
        switch discretizer.encode {
        case .ordinal:
            for (colIdx, colName) in names.enumerated() {
                let colValues = (0..<shape.rows).map { rowIdx in binned[rowIdx][colIdx] }
                let newCol = TypedColumn<Double>(name: colName, values: colValues)
                df = try df.withColumn(colName, column: newCol)
            }
        case .oneHot:
            df = try df.drop(names)
            for (colIdx, colName) in names.enumerated() {
                for bin in 0..<discretizer.nBins {
                    let binColName = "\(colName)_bin_\(bin)"
                    let colValues = (0..<shape.rows).map { rowIdx in
                        binned[rowIdx][colIdx * discretizer.nBins + bin]
                    }
                    let binCol = TypedColumn<Double>(name: binColName, values: colValues)
                    df = try df.withColumn(binColName, column: binCol)
                }
            }
        }
        
        return df
    }
    
    /// Fits and discretizes the specified columns, returning the discretized DataFrame and discretizer.
    public func kBinsDiscretize(columns names: [String], nBins: Int = 5, strategy: KBinsDiscretizer.Strategy = .uniform, encode: KBinsDiscretizer.Encode = .ordinal) throws -> (discretized: DataFrame, discretizer: KBinsDiscretizer) {
        let discretizer = try fitKBinsDiscretizer(columns: names, nBins: nBins, strategy: strategy, encode: encode)
        let discretizedDf = try kBinsDiscretize(columns: names, discretizer: discretizer)
        return (discretizedDf, discretizer)
    }
}
