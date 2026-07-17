import Testing
import Foundation
import MLX
@testable import SwiftLLM

@Suite("SafeTensors and GGUF Parser Tests")
struct ParserTests {
    
    @Test("SafeTensors Parser reads mock file")
    func testSafeTensorsParsing() throws {
        // Create mock SafeTensors content
        let headerJSON = """
        {
          "weight1": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
          },
          "weight2": {
            "dtype": "F16",
            "shape": [4],
            "data_offsets": [24, 32]
          }
        }
        """
        
        let headerData = headerJSON.data(using: .utf8)!
        let headerLen = UInt64(headerData.count)
        
        var fileData = Data()
        var lenBytes = headerLen.littleEndian
        withUnsafeBytes(of: &lenBytes) { bytes in
            fileData.append(contentsOf: bytes)
        }
        fileData.append(headerData)
        
        let w1Data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        w1Data.withUnsafeBytes { bytes in
            fileData.append(contentsOf: bytes)
        }
        
        let w2Data: [Float16] = [10.0, 20.0, 30.0, 40.0]
        w2Data.withUnsafeBytes { bytes in
            fileData.append(contentsOf: bytes)
        }
        
        let fm = FileManager.default
        let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)
        let fileURL = currentDir.appendingPathComponent("test_mock_weights.safetensors")
        
        try? fm.removeItem(at: fileURL)
        defer { try? fm.removeItem(at: fileURL) }
        
        try fileData.write(to: fileURL)
        
        let tensors = try SafeTensorsParser.parse(url: fileURL)
        
        #expect(tensors.count == 2)
        #expect(tensors["weight1"]?.shape == [2, 3])
        #expect(tensors["weight2"]?.shape == [4])
        
        if let w1 = tensors["weight1"] {
            let vals = w1.asArray(Float.self)
            #expect(vals == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        }
    }
    
    @Test("GGUF Parser reads mock file")
    func testGGUFParsing() throws {
        var fileData = Data()
        
        // 1. Magic
        fileData.append(Data([0x47, 0x47, 0x55, 0x46]))
        // 2. Version
        var version = UInt32(3).littleEndian
        withUnsafeBytes(of: &version) { bytes in
            fileData.append(contentsOf: bytes)
        }
        // 3. Tensor count (1)
        var tensorCount = UInt64(1).littleEndian
        withUnsafeBytes(of: &tensorCount) { bytes in
            fileData.append(contentsOf: bytes)
        }
        // 4. Metadata KV count (0)
        var metadataCount = UInt64(0).littleEndian
        withUnsafeBytes(of: &metadataCount) { bytes in
            fileData.append(contentsOf: bytes)
        }
        
        // 5. Tensor info
        let name = "layer1.weight"
        var nameLen = UInt64(name.utf8.count).littleEndian
        withUnsafeBytes(of: &nameLen) { bytes in
            fileData.append(contentsOf: bytes)
        }
        fileData.append(name.data(using: .utf8)!)
        
        // Dimensions count: 2
        var dimsCount = UInt32(2).littleEndian
        withUnsafeBytes(of: &dimsCount) { bytes in
            fileData.append(contentsOf: bytes)
        }
        // Dimensions: 2, 3
        var dim1 = UInt64(2).littleEndian
        withUnsafeBytes(of: &dim1) { bytes in
            fileData.append(contentsOf: bytes)
        }
        var dim2 = UInt64(3).littleEndian
        withUnsafeBytes(of: &dim2) { bytes in
            fileData.append(contentsOf: bytes)
        }
        
        // Tensor type: 0 (F32)
        var tensorType = UInt32(0).littleEndian
        withUnsafeBytes(of: &tensorType) { bytes in
            fileData.append(contentsOf: bytes)
        }
        // Tensor offset: 0
        var tensorOffset = UInt64(0).littleEndian
        withUnsafeBytes(of: &tensorOffset) { bytes in
            fileData.append(contentsOf: bytes)
        }
        
        // Align binary block start to 32 bytes boundary
        let currentOffset = fileData.count
        let alignment = 32
        let binaryStart = (currentOffset + alignment - 1) & ~(alignment - 1)
        let paddingSize = binaryStart - currentOffset
        if paddingSize > 0 {
            fileData.append(Data(repeating: 0, count: paddingSize))
        }
        
        // Binary tensor data (6 Float32 values = 24 bytes)
        let tensorBytes: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        tensorBytes.withUnsafeBytes { bytes in
            fileData.append(contentsOf: bytes)
        }
        
        let fm = FileManager.default
        let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)
        let fileURL = currentDir.appendingPathComponent("test_mock_weights.gguf")
        
        try? fm.removeItem(at: fileURL)
        defer { try? fm.removeItem(at: fileURL) }
        
        try fileData.write(to: fileURL)
        
        let tensors = try GGUFParser.parse(url: fileURL)
        #expect(tensors.count == 1)
        #expect(tensors["layer1.weight"]?.shape == [2, 3])
        
        if let w = tensors["layer1.weight"] {
            let vals = w.asArray(Float.self)
            #expect(vals == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        }
    }
}
