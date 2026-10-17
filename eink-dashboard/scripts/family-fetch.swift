// Family Memory (DECISIONS #92): download one Photos library picture, upright, about 1600 px on its
// long side, through PhotoKit. iCloud-only originals (the shared album keeps only previews on the
// Mac) are fetched over the network at that size; nothing else is synced.
//
//   family-fetch <localIdentifier> <out.jpg>     exit 0 = written
//
// Built into a tiny app, so macOS asks for Photos access once, on screen, and remembers it for the app
// (a launchd job cannot show that question): scripts/build-family-fetch.sh. The job launches it with
// `open -W`, and the result (the picture's size, or the error) goes to <out.jpg>.txt.
import AppKit
import Photos

let args = CommandLine.arguments
guard args.count == 3 else { FileHandle.standardError.write("usage: family-fetch <localIdentifier> <out.jpg>\n".data(using: .utf8)!); exit(2) }
let report = args[2] + ".txt"
func finish(_ text: String, _ code: Int32) -> Never { try? text.write(toFile: report, atomically: true, encoding: .utf8); exit(code) }
var status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
if status == .notDetermined {
  let wait = DispatchSemaphore(value: 0)
  PHPhotoLibrary.requestAuthorization(for: .readWrite) { s in status = s; wait.signal() }
  wait.wait()
}
guard status == .authorized || status == .limited else { finish("no Photos access (status \(status.rawValue))", 3) }
guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [args[1]], options: nil).firstObject else { finish("not found", 4) }

let opts = PHImageRequestOptions()
opts.isNetworkAccessAllowed = true
opts.isSynchronous = true
opts.deliveryMode = .highQualityFormat
opts.resizeMode = .exact
var result = ("no image", Int32(5))
PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 1600, height: 1600), contentMode: .aspectFit, options: opts) { image, info in
  guard let image = image, let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
        let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.88]) else {
    result = ("no image: \(String(describing: info?[PHImageErrorKey]))", 5)
    return
  }
  do { try jpeg.write(to: URL(fileURLWithPath: args[2])); result = ("\(rep.pixelsWide)x\(rep.pixelsHigh)", 0) }
  catch { result = ("write failed: \(error)", 6) }
}
finish(result.0, result.1)
