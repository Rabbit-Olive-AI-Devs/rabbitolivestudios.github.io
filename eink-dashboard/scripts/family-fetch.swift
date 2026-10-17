// Family Memory (DECISIONS #92): download one Photos library picture, upright, about 1600 px on its
// long side, through PhotoKit. iCloud-only originals (the shared album keeps only previews on the
// Mac) are fetched over the network at that size; nothing else is synced.
//
//   family-fetch <localIdentifier> <out.jpg>     exit 0 = written
//
// Compiled once to ~/Library/Application Support/eink-family/family-fetch, so the Photos privacy
// permission attaches to that one binary (the launchd job runs it):
//   swiftc -O scripts/family-fetch.swift -o "$HOME/Library/Application Support/eink-family/family-fetch"
import AppKit
import Photos

let args = CommandLine.arguments
guard args.count == 3 else { FileHandle.standardError.write("usage: family-fetch <localIdentifier> <out.jpg>\n".data(using: .utf8)!); exit(2) }
let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
guard status == .authorized || status == .limited else { FileHandle.standardError.write("no Photos access (status \(status.rawValue))\n".data(using: .utf8)!); exit(3) }
guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [args[1]], options: nil).firstObject else { FileHandle.standardError.write("not found\n".data(using: .utf8)!); exit(4) }

let opts = PHImageRequestOptions()
opts.isNetworkAccessAllowed = true
opts.isSynchronous = true
opts.deliveryMode = .highQualityFormat
opts.resizeMode = .exact
var code: Int32 = 5
PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 1600, height: 1600), contentMode: .aspectFit, options: opts) { image, info in
  guard let image = image, let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
        let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.88]) else {
    FileHandle.standardError.write("no image: \(String(describing: info?[PHImageErrorKey]))\n".data(using: .utf8)!)
    return
  }
  do { try jpeg.write(to: URL(fileURLWithPath: args[2])); print("\(rep.pixelsWide)x\(rep.pixelsHigh)"); code = 0 }
  catch { FileHandle.standardError.write("write failed: \(error)\n".data(using: .utf8)!) }
}
exit(code)
