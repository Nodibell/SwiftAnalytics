// swift-tools-version: 6.0

import PackageDescription

let globalSwiftSettings: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny")
]

let globalCSettings: [CSetting] = [
    .define("ACCELERATE_NEW_LAPACK"),
    .define("ACCELERATE_LAPACK_ILP64")
]

let package = Package(
    name: "SwiftAnalytics",
    platforms: [
        .macOS(.v14),
        .iOS(.v18),
        .visionOS(.v2),
    ],
    products: [
        .library(name: "SwiftDataFrame", targets: ["SwiftDataFrame"]),
        .library(name: "SwiftStats",     targets: ["SwiftStats"]),
        .library(name: "SwiftPreprocessing", targets: ["SwiftPreprocessing"]),
        .library(name: "SwiftML",           targets: ["SwiftML"]),
        .library(name: "SwiftCluster",       targets: ["SwiftCluster"]),
        .library(name: "SwiftNLP",           targets: ["SwiftNLP"]),
        .library(name: "SwiftOptimize",      targets: ["SwiftOptimize"]),
        .library(name: "SwiftForecast",      targets: ["SwiftForecast"]),
        .library(name: "SwiftLLM",           targets: ["SwiftLLM"]),
        .library(name: "SwiftExplain",       targets: ["SwiftExplain"]),
        .library(name: "SwiftPrivacy",       targets: ["SwiftPrivacy"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apache/arrow-swift.git",
            from: "21.0.0"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.31.6"
        ),
    ],
    targets: [
        // ── SwiftDataFrame ──────────────────────────────────────────────
        .target(
            name: "SwiftDataFrame",
            dependencies: [
                .product(name: "Arrow", package: "arrow-swift")
            ],
            path: "Sources/SwiftDataFrame",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftDataFrameTests",
            dependencies: ["SwiftDataFrame"],
            path: "Tests/SwiftDataFrameTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftStats ───────────────────────────────────────────────────
        .target(
            name: "SwiftStats",
            dependencies: ["SwiftDataFrame"],
            path: "Sources/SwiftStats",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]

        ),
        .testTarget(
            name: "SwiftStatsTests",
            dependencies: ["SwiftStats"],
            path: "Tests/SwiftStatsTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftPreprocessing ────────────────────────────────────────────
        .target(
            name: "SwiftPreprocessing",
            dependencies: [
                "SwiftDataFrame",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/SwiftPreprocessing",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftPreprocessingTests",
            dependencies: ["SwiftPreprocessing"],
            path: "Tests/SwiftPreprocessingTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftML ──────────────────────────────────────────────────────
        .target(
            name: "SwiftML",
            dependencies: [
                "SwiftPreprocessing",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/SwiftML",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftMLTests",
            dependencies: ["SwiftML"],
            path: "Tests/SwiftMLTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftCluster ─────────────────────────────────────────────────
        .target(
            name: "SwiftCluster",
            dependencies: [
                "SwiftDataFrame",
                "SwiftPreprocessing",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/SwiftCluster",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "SwiftClusterTests",
            dependencies: ["SwiftCluster", "SwiftPreprocessing"],
            path: "Tests/SwiftClusterTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftNLP ─────────────────────────────────────────────────────
        .target(
            name: "SwiftNLP",
            dependencies: [
                "SwiftDataFrame",
            ],
            path: "Sources/SwiftNLP",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftNLPTests",
            dependencies: ["SwiftNLP"],
            path: "Tests/SwiftNLPTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftOptimize ────────────────────────────────────────────────
        .target(
            name: "SwiftOptimize",
            dependencies: [
                "SwiftDataFrame",
                "SwiftML",
            ],
            path: "Sources/SwiftOptimize",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftOptimizeTests",
            dependencies: ["SwiftOptimize"],
            path: "Tests/SwiftOptimizeTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftForecast ────────────────────────────────────────────────
        .target(
            name: "SwiftForecast",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
            ],
            path: "Sources/SwiftForecast",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "SwiftForecastTests",
            dependencies: ["SwiftForecast"],
            path: "Tests/SwiftForecastTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftLLM ─────────────────────────────────────────────────────
        .target(
            name: "SwiftLLM",
            dependencies: [
                "SwiftNLP",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/SwiftLLM",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftLLMTests",
            dependencies: ["SwiftLLM"],
            path: "Tests/SwiftLLMTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftExplain ─────────────────────────────────────────────────
        .target(
            name: "SwiftExplain",
            dependencies: [
                "SwiftML",
                "SwiftStats",
                "SwiftDataFrame",
                "SwiftPreprocessing",
            ],
            path: "Sources/SwiftExplain",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftExplainTests",
            dependencies: ["SwiftExplain"],
            path: "Tests/SwiftExplainTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftPrivacy ─────────────────────────────────────────────────
        .target(
            name: "SwiftPrivacy",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
            ],
            path: "Sources/SwiftPrivacy",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
        .testTarget(
            name: "SwiftPrivacyTests",
            dependencies: ["SwiftPrivacy"],
            path: "Tests/SwiftPrivacyTests",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),

        // ── SwiftAnalyticsBenchmarks ──────────────────────────────────────
        .executableTarget(
            name: "SwiftAnalyticsBenchmarks",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
                "SwiftPreprocessing",
                "SwiftML",
                "SwiftCluster",
                "SwiftNLP",
                "SwiftOptimize",
                "SwiftForecast",
                "SwiftLLM",
                "SwiftExplain",
                "SwiftPrivacy",
            ],
            path: "Benchmarks/Swift",
            swiftSettings: globalSwiftSettings,
            cSettings: globalCSettings
        ),
    ]
)
