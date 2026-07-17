import Foundation

/// Target compute device for estimators that support hardware routing.
public enum ExecutionDevice: String, Sendable, Codable, Equatable {
    case cpu
    case gpu
    /// Reserved for CoreML / Apple Neural Engine (falls back until SwiftLLM).
    case ane
    /// Choose CPU or GPU from data size and algorithm heuristics.
    case auto
}
