// Family Memory (DECISIONS #92): download one Photos library picture, upright, about 1600 px on its
// long side, through PhotoKit. iCloud-only originals (the shared album keeps only previews on the
// Mac) are fetched over the network at that size; nothing else is synced.
//
//   family-fetch <localIdentifier> <out.jpg>     exit 0 = written (legacy)
//   family-fetch --select-and-fetch YYYY-MM-DD requestUUID
// The unattended mode never prompts; full authorization and successful DB reads are separate gates.
//
// Built into a tiny app, so macOS asks for Photos access once, on screen, and remembers it for the app
// (a launchd job cannot show that question): scripts/build-family-fetch.sh. The job launches it with
// `open -W`, and the result (the picture's size, or the error) goes to <out.jpg>.txt.
import AppKit
import Photos

@main enum FamilyFetch {
  static func main() {
    let args = CommandLine.arguments
    if args.dropFirst().first == "--select-and-fetch" {
      guard args.count == 4, FamilyCore.validRequest(args[2], args[3]) else { blocked(2) }
      do {
        let lease = try FamilyOwnership(home:FileManager.default.homeDirectoryForCurrentUser,day:args[2],request:args[3])
        // Registration/control is non-Photos metadata. No consent request, config or roster read.
        do {
          try lease.check()
          // No config, DB, roster, shown log, or PhotoKit fetch before the FULL-access guard.
          try FamilyCore.authorized(PHPhotoLibrary.authorizationStatus(for: .readWrite) == .authorized) {
            try selectAndFetch(day:args[2], request:args[3], lease:lease)
          }
          try lease.finish(status:"complete")
          withExtendedLifetime(lease) { exit(0) }
        } catch {
          // Failed private cleanup cannot be acknowledged as stopped/clean. Retain the owner.
          if case FamilyFailure.cleanupIncomplete = error { withExtendedLifetime(lease) { blocked(3) } }
          let status = PHPhotoLibrary.authorizationStatus(for:.readWrite)
          let authorization:String?
          switch status {
            case .notDetermined: authorization="notDetermined"
            case .limited: authorization="limited"
            case .denied: authorization="denied"
            case .restricted: authorization="restricted"
            default: authorization=nil
          }
          try lease.finish(status:authorization == nil ? "blocked" : "full-Photos-access-required",authorization:authorization)
          withExtendedLifetime(lease) { blocked(3) }
        }
      } catch { blocked(3) }
    }
    legacy(args)
  }
  static func blocked(_ code: Int32) -> Never {
    FileHandle.standardOutput.write(Data("{\"status\":\"blocked\"}\n".utf8))
    exit(code)
  }
  static func selectAndFetch(day: String, request: String, lease: FamilyOwnership) throws {
    try lease.check()
    let fm = FileManager.default
    let home = fm.homeDirectoryForCurrentUser
    let config = home.appendingPathComponent(".config/eink/paths.json")
    try FamilyFiles.safe(config)
    var settings: [String:Any] = [:]
    if try FamilyFiles.exists(config) {
      guard let json = try JSONSerialization.jsonObject(with:Data(contentsOf:config)) as? [String:Any] else { throw FamilyFailure.blocked }
      settings = json
    }
    func configured(_ key: String, _ fallback: String) throws -> URL {
      if let value = settings[key], !(value is NSNull) {
        guard let path = value as? String, path.hasPrefix("/"), !path.contains("/../"), !path.contains("/./") else { throw FamilyFailure.blocked }
        return URL(fileURLWithPath:path, isDirectory:true)
      }
      return URL(fileURLWithPath:fallback, isDirectory:true)
    }
    let library = try configured("photosLibrary", "/Volumes/SamsungSSD/Pictures/Photos Library.photoslibrary")
    let output = try configured("familyTmp", "/Volumes/SamsungSSD/Pictures/.eink-family-tmp")
    try lease.validateOutput(output)
    let cache = home.appendingPathComponent("Library/Caches/com.eink.family-fetch", isDirectory:true)
    try FamilyFiles.safe(library); try FamilyFiles.safe(output)
    // The output root cannot be a library, home/config/cache, or one of their ancestors.
    for protected in [library, home.appendingPathComponent(".config"), cache] {
      guard output.path != protected.path, !output.path.hasPrefix(protected.path+"/"),
        !protected.path.hasPrefix(output.path+"/") else { throw FamilyFailure.blocked }
    }
    guard output.path != home.path else { throw FamilyFailure.blocked }
    guard let roster = Bundle.main.url(forResource: "people", withExtension: "json") else { throw FamilyFailure.blocked }
    try FamilyFiles.safe(roster)
    let people = try JSONDecoder().decode([String:String].self, from:Data(contentsOf:roster))
    let shownURL = home.appendingPathComponent(".config/eink/family-shown.json")
    try FamilyFiles.safe(shownURL)
    let shown = try FamilyFiles.exists(shownURL)
      ? JSONDecoder().decode([FamilyShown].self, from:Data(contentsOf:shownURL)) : []
    try lease.check()
    let (round,candidates) = try FamilyFiles.withSnapshot(library:library, cache:cache) { db in
      try lease.check()
      let (all,_) = try FamilyCore.readCandidates(db.path, people:people)
      try lease.check()
      return try FamilyCore.choose(all, shown:shown, target:day)
    }
    try lease.check()
    // Raw snapshot and SQLite handles are gone before any winner bytes are published.
    guard let (candidate,jpeg,width,height) = try FamilyCore.fetchWinner(candidates, download: { candidate in
      guard PHPhotoLibrary.authorizationStatus(for:.readWrite) == .authorized else { throw FamilyFailure.blocked }
      guard UUID(uuidString:candidate.uuid) != nil else { throw FamilyFailure.blocked }
      try lease.check()
      let image = try download(candidate.uuid + "/L0/001")
      try lease.check()
      return image
    }) else { throw FamilyFailure.noImage }
    let receipt = FamilyReceipt(request:request, date:day, uuid:candidate.uuid, taken:candidate.taken,
      source:candidate.source, round:round, width:width, height:height)
    try lease.check()
    try FamilyFiles.publish(jpeg, receipt:receipt, root:output)
    do { try lease.check() } catch {
      let folder=output.appendingPathComponent(request,isDirectory:true)
      do { try FamilyFiles.safe(folder); try fm.removeItem(at:folder) }
      catch { throw FamilyFailure.cleanupIncomplete }
      throw FamilyFailure.blocked
    }
  }
  static func download(_ identifier: String) throws -> (Data,Int,Int)? {
    guard let asset = PHAsset.fetchAssets(withLocalIdentifiers:[identifier], options:nil).firstObject else {
      guard PHPhotoLibrary.authorizationStatus(for:.readWrite) == .authorized else { throw FamilyFailure.blocked }
      return nil
    }
    let opts = PHImageRequestOptions()
    opts.isNetworkAccessAllowed = true
    opts.isSynchronous = true
    opts.deliveryMode = .highQualityFormat
    opts.resizeMode = .exact
    var result: (Data,Int,Int)?
    var denied = false
    PHImageManager.default().requestImage(for:asset, targetSize:CGSize(width:1600,height:1600), contentMode:.aspectFit, options:opts) { image, info in
      if let error = info?[PHImageErrorKey] as? NSError, FamilyCore.protectedDenial(error) { denied = true; return }
      guard let image = image, let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data:tiff),
        let jpeg = rep.representation(using:.jpeg, properties:[.compressionFactor:0.88]) else { return }
      result = (jpeg,rep.pixelsWide,rep.pixelsHigh)
    }
    guard !denied, PHPhotoLibrary.authorizationStatus(for:.readWrite) == .authorized else { throw FamilyFailure.blocked }
    return result
  }
  // Original legacy download mode: retains its interactive/limited authorization behavior and .txt report.
  static func legacy(_ args: [String]) -> Never {
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
  }
}
