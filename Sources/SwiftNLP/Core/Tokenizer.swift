import Foundation
@_exported import SwiftDataFrame

/// A unified protocol for tokenizing and encoding text.
public protocol Tokenizer: Sendable {
    /// Splits the given text into an array of string tokens.
    /// - Parameter text: The input string.
    /// - Returns: An array of token strings.
    func tokenize(text: String) -> [String]
    
    /// Encodes the given text into a sequence of token IDs.
    /// - Parameter text: The input string.
    /// - Returns: An array of token ID integers.
    func encode(text: String) -> [Int]
    
    /// Decodes a sequence of token IDs back into a reconstructed string.
    /// - Parameter tokens: An array of token ID integers.
    /// - Returns: The reconstructed string.
    func decode(tokens: [Int]) -> String
}
