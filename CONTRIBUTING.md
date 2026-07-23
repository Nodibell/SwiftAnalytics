# Contributing to SwiftSci

Thank you for your interest in contributing to **SwiftSci**! SwiftSci is a high-performance, modular Swift 6 data science and machine learning framework designed for Apple Silicon.

## Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.

## Getting Started

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/swiftsci/SwiftSci.git
   cd SwiftSci
   ```

2. **Build and Test:**
   ```bash
   swift build
   swift test
   ```

3. **Check Swift 6 Concurrency:**
   Ensure all target modifications adhere strictly to Swift 6 strict concurrency checks without warnings or data races (`@Sendable`, `actor`, thread-safe data structures).

## Development Guidelines

- **Architecture:** Keep target responsibilities clean across `SwiftDataFrame`, `SwiftStats`, `SwiftML`, `SwiftCluster`, `SwiftNLP`, `SwiftForecast`, `SwiftLLM`, etc.
- **Performance:** Leverage Apple Accelerate (`vDSP`, `LAPACK`) or `MLX` where appropriate for vectorized compute.
- **Documentation:** Every public API method, struct, enum, and class must be documented using Swift DocC comments (`///`).
- **Tests:** Add unit tests under `Tests/<Target>Tests` for every new algorithm or feature.

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/my-new-algorithm`).
2. Verify all unit tests pass (`swift test`).
3. Commit changes with clear, descriptive commit messages.
4. Push to your branch and open a Pull Request targeting `main`.

Thank you for helping make Swift data science robust, fast, and native!
