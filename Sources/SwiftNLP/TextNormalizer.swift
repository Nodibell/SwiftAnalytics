import Foundation

/// Utilities for standard text normalization.
public struct TextNormalizer: Sendable {
    public var lowercase: Bool
    public var removePunctuation: Bool

    public init(lowercase: Bool = true, removePunctuation: Bool = true) {
        self.lowercase = lowercase
        self.removePunctuation = removePunctuation
    }

    /// Normalizes input text according to configured options and Unicode normalization.
    public func normalize(_ text: String) -> String {
        var result = text.precomposedStringWithCanonicalMapping
        if lowercase {
            result = result.lowercased()
        }
        if removePunctuation {
            result = result.components(separatedBy: .punctuationCharacters).joined()
        }
        return result
    }
}
