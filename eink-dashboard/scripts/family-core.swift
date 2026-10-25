// Selection parity with the immutable family-pick.py. No PhotoKit, subprocess or CLI overrides.
import Foundation
import SQLite3
import Darwin

// Fixed request-control metadata only. No Photos/config/roster reads in this protocol.
// A stalled/crashed process is NEVER inferred stopped from the open(1) waiter.
final class FamilyOwnership {
  private struct Owner: Decodable { let request:String; let date:String; let pid:Int32; let deadline:Double; let tmp:String }
  private let active:URL
  private let day:String
  private let request:String
  private let owner:Owner
  private var fd:Int32 = -1
  init(home:URL, day:String, request:String) throws {
    guard FamilyCore.validRequest(day,request) else { throw FamilyFailure.blocked }
    self.day=day;self.request=request
    let root=home.appendingPathComponent("Library/Caches/com.eink.family-fetch-control",isDirectory:true)
    active=root.appendingPathComponent("active",isDirectory:true)
    try FamilyFiles.safe(active)
    guard try FamilyFiles.exists(active) else { throw FamilyFailure.blocked }
    let attributes=try FileManager.default.attributesOfItem(atPath:active.path)
    guard attributes[.type] as? FileAttributeType == .typeDirectory,
      (attributes[.ownerAccountID] as? NSNumber)?.uint32Value==getuid() else { throw FamilyFailure.blocked }
    let ownerURL=active.appendingPathComponent("owner.json")
    try FamilyFiles.safe(ownerURL)
    owner=try JSONDecoder().decode(Owner.self,from:Data(contentsOf:ownerURL))
    guard owner.request==request,owner.date==day,owner.pid>1,owner.deadline.isFinite else { throw FamilyFailure.blocked }
    // Persistent inode: Node never deletes this gate, even during request cleanup.
    let gate=root.appendingPathComponent("producer.lock")
    try FamilyFiles.safe(gate)
    fd=open(gate.path,O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC,0o600)
    guard fd >= 0 else { throw FamilyFailure.blocked }
    var st=stat()
    guard fstat(fd,&st)==0,st.st_uid==getuid(),(st.st_mode & S_IFMT)==S_IFREG,
      flock(fd,LOCK_EX|LOCK_NB)==0 else { close(fd);fd = -1;throw FamilyFailure.blocked }
    do { try write("registered.json",["request":request,"date":day,"pid":getpid()]) }
    catch { close(fd);fd = -1;throw FamilyFailure.blocked }
  }
  deinit { if fd >= 0 { close(fd) } }
  private func bound() throws {
    try FamilyFiles.safe(active.appendingPathComponent("owner.json"))
    let current=try JSONDecoder().decode(Owner.self,from:Data(contentsOf:active.appendingPathComponent("owner.json")))
    guard current.request==request,current.date==day,current.pid==owner.pid,current.deadline==owner.deadline,current.tmp==owner.tmp else { throw FamilyFailure.blocked }
  }
  func check() throws {
    try bound()
    let now=Date().timeIntervalSince1970*1000
    guard try !FamilyFiles.exists(active.appendingPathComponent("cancelled")),now<owner.deadline,
      owner.deadline<=now+180000,kill(owner.pid,0)==0 else { throw FamilyFailure.blocked }
  }
  func validateOutput(_ output:URL) throws {
    try check()
    guard owner.tmp.hasPrefix("/"),URL(fileURLWithPath:owner.tmp,isDirectory:true).path==output.path else { throw FamilyFailure.blocked }
  }
  private func write(_ name:String,_ value:[String:Any]) throws {
    try bound();let target=active.appendingPathComponent(name);try FamilyFiles.safe(target)
    try JSONSerialization.data(withJSONObject:value).write(to:target,options:.atomic)
    try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:target.path)
  }
  // Terminal acknowledgment is a promise: all protected operations/cleanup have finished.
  // The producer lock is still held; no private work is allowed after this write.
  func finish(status:String,authorization:String?=nil) throws {
    guard ["complete","cancelled","blocked","full-Photos-access-required"].contains(status) else { throw FamilyFailure.blocked }
    var value:[String:Any]=["request":request,"date":day,"pid":getpid(),"status":status]
    if let authorization=authorization {
      guard ["notDetermined","limited","denied","restricted"].contains(authorization) else { throw FamilyFailure.blocked }
      value["authorization"]=authorization
    }
    try write("stopped.json",value)
  }
}

// Small fixed-purpose snapshot boundary. Only the app calls this in production.
enum FamilyFiles {
  static let fm = FileManager.default
  static func exists(_ url: URL) throws -> Bool {
    var st = stat()
    if lstat(url.path, &st) == 0 { return true }
    if errno == ENOENT { return false }
    throw FamilyFailure.blocked
  }
  static func safe(_ url: URL) throws {
    guard url.path.hasPrefix("/"), url.pathComponents.count >= 3,
      url.standardizedFileURL.path == url.path else { throw FamilyFailure.blocked }
    var p = URL(fileURLWithPath:"/", isDirectory:true)
    for part in url.pathComponents.dropFirst() {
      p.appendPathComponent(part)
      if try exists(p) {
        let attrs = try fm.attributesOfItem(atPath:p.path)
        guard attrs[.type] as? FileAttributeType != .typeSymbolicLink else { throw FamilyFailure.blocked }
      }
    }
  }
  static func privateDirectory(_ url: URL) throws {
    try safe(url)
    try fm.createDirectory(at:url, withIntermediateDirectories:true, attributes:[.posixPermissions:0o700])
    let attrs = try fm.attributesOfItem(atPath:url.path)
    guard attrs[.type] as? FileAttributeType == .typeDirectory,
      (attrs[.ownerAccountID] as? NSNumber)?.uint32Value == getuid() else { throw FamilyFailure.blocked }
    try fm.setAttributes([.posixPermissions:0o700], ofItemAtPath:url.path)
  }
  static func publish(_ jpeg: Data, receipt: FamilyReceipt, root: URL) throws {
    guard FamilyCore.validRequest(receipt.date, receipt.request), UUID(uuidString:receipt.uuid) != nil,
      FamilyCore.date(receipt.taken) != nil, ["album","favourite"].contains(receipt.source),
      receipt.round >= 1, receipt.width >= 1024, receipt.height > 0 else { throw FamilyFailure.invalid }
    try privateDirectory(root)
    let folder = root.appendingPathComponent(receipt.request, isDirectory:true)
    guard try !exists(folder) else { throw FamilyFailure.blocked }
    try fm.createDirectory(at:folder, withIntermediateDirectories:false, attributes:[.posixPermissions:0o700])
    do {
      try jpeg.write(to:folder.appendingPathComponent("photo.jpg"), options:.atomic)
      try JSONEncoder().encode(receipt).write(to:folder.appendingPathComponent("selection.json"), options:.atomic)
      let complete = try JSONSerialization.data(withJSONObject:["request":receipt.request,"date":receipt.date])
      try complete.write(to:folder.appendingPathComponent("completed.json"), options:.atomic)
    } catch {
      do { try fm.removeItem(at:folder) } catch { throw FamilyFailure.cleanupIncomplete }
      throw FamilyFailure.blocked
    }
  }
  static func withSnapshot<T>(library: URL, cache: URL, body: (URL) throws -> T) throws -> T {
    try safe(library); try safe(cache)
    guard !cache.path.hasPrefix(library.path + "/"), cache.path != library.path,
      !library.path.hasPrefix(cache.path + "/") else { throw FamilyFailure.blocked }
    let source = library.appendingPathComponent("database", isDirectory:true)
    try safe(source)
    try privateDirectory(cache)
    let scratch = cache.appendingPathComponent(UUID().uuidString, isDirectory:true)
    try fm.createDirectory(at:scratch, withIntermediateDirectories:false, attributes:[.posixPermissions:0o700])
    do {
      for name in ["Photos.sqlite","Photos.sqlite-wal","Photos.sqlite-shm"] {
        let file = source.appendingPathComponent(name)
        guard try exists(file) else {
          if name == "Photos.sqlite" { throw FamilyFailure.blocked }
          continue
        }
        try safe(file)
        guard try fm.attributesOfItem(atPath:file.path)[.type] as? FileAttributeType == .typeRegular else { throw FamilyFailure.blocked }
        try fm.copyItem(at:file, to:scratch.appendingPathComponent(name))
        try fm.setAttributes([.posixPermissions:0o600], ofItemAtPath:scratch.appendingPathComponent(name).path)
      }
      let result = try body(scratch.appendingPathComponent("Photos.sqlite"))
      try fm.removeItem(at:scratch) // Cleanup must succeed before selected output may be published.
      return result
    } catch {
      do { try fm.removeItem(at:scratch) } catch { throw FamilyFailure.cleanupIncomplete }
      throw FamilyFailure.blocked
    }
  }
}

struct FamilyCandidate: Codable {
  let uuid: String
  let taken: String
  let source: String
  let pixels: Int64
}
struct FamilyShown: Codable { let uuid: String; let date: String; let round: Int }
struct FamilyReceipt: Codable {
  let request: String; let date: String; let uuid: String; let taken: String; let source: String
  let round: Int; let width: Int; let height: Int
}
enum FamilyFailure: Error { case blocked, invalid, noImage, cleanupIncomplete }

enum FamilyCore {
  static func fetchWinner(_ candidates: [FamilyCandidate], download: (FamilyCandidate) throws -> (Data,Int,Int)?) throws -> (FamilyCandidate,Data,Int,Int)? {
    for candidate in candidates {
      guard let (jpeg,width,height) = try download(candidate), width >= 1024 else { continue }
      return (candidate,jpeg,width,height)
    }
    return nil
  }
  static func protectedDenial(_ error: NSError, depth: Int = 0) -> Bool {
    if error.domain == "PHPhotosErrorDomain" && [3310,3311].contains(error.code) { return true }
    if error.domain == NSCocoaErrorDomain && [257,513].contains(error.code) { return true }
    if error.domain == NSPOSIXErrorDomain && [Int(EPERM),Int(EACCES)].contains(error.code) { return true }
    if depth < 8, let underlying = error.userInfo[NSUnderlyingErrorKey] as? NSError {
      return protectedDenial(underlying, depth:depth+1)
    }
    return false
  }
  static func leap(_ y: Int) -> Bool { y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) }
  static func monthDays(_ y: Int) -> [Int] { [31,leap(y) ? 29 : 28,31,30,31,30,31,31,30,31,30,31] }
  static func ordinal(_ year: Int, _ month: Int, _ day: Int) -> Int {
    let y = year-1
    return 365*y + y/4 - y/100 + y/400 + monthDays(year).prefix(month-1).reduce(0,+) + day
  }
  static func date(_ text: String) -> Date? {
    guard text.range(of: #"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"#, options:.regularExpression) != nil else { return nil }
    let p = text.split(separator:"-").compactMap { Int($0) }
    guard p.count == 3, p[0] > 0, (1...12).contains(p[1]),
      (1...monthDays(p[0])[p[1]-1]).contains(p[2]) else { return nil }
    // Python uses a proleptic Gregorian calendar; Foundation defaults to a 1582 cutover.
    return Date(timeIntervalSince1970:Double(ordinal(p[0],p[1],p[2])-719163)*86400)
  }
  static func iso(_ d: Date) -> String {
    let value = floor(d.timeIntervalSince1970/86400)+719163
    guard value.isFinite, value >= 1, value <= 3652059 else { return "" }
    let day = Int(value)
    var y = min(9999,max(1,Int(Double(day)/365.2425)+1))
    while ordinal(y,1,1) > day { y -= 1 }
    while ordinal(y+1,1,1) <= day { y += 1 }
    var left = day-ordinal(y,1,1)+1
    var m = 1
    for days in monthDays(y) { if left <= days { break }; left -= days; m += 1 }
    return String(format:"%04d-%02d-%02d",y,m,left)
  }
  static func validRequest(_ day: String, _ request: String) -> Bool {
    date(day) != nil && request.range(of: #"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"#, options:.regularExpression) != nil && UUID(uuidString: request) != nil
  }
  static func authorized<T>(_ fullAccess: Bool, metadata: () throws -> T) throws -> T {
    guard fullAccess else { throw FamilyFailure.blocked }
    return try metadata()
  }
  static func distance(_ a: Date, _ b: Date) -> Int {
    func leapOrdinal(_ d: Date) -> Int {
      let parts = iso(d).split(separator:"-").compactMap { Int($0) }
      return ordinal(2000,parts[1],parts[2])-ordinal(2000,1,1)+1
    }
    let delta = abs(leapOrdinal(a)-leapOrdinal(b))
    return min(delta,366-delta)
  }
  static func choose(_ candidates: [FamilyCandidate], shown: [FamilyShown], target: String) throws -> (Int,[FamilyCandidate]) {
    guard let day = date(target) else { throw FamilyFailure.invalid }
    let albumIDs = Set(candidates.filter { $0.source == "album" }.map { $0.uuid })
    var round = shown.map { $0.round }.max() ?? 1
    var seen = Set(shown.filter { $0.round == round }.map { $0.uuid })
    if !albumIDs.isEmpty && albumIDs.isSubset(of: seen) { round += 1; seen = [] }
    // Original input position explicitly breaks ties, matching Python's stable sorted().
    var ranked: [(Int,Int,Int64,Int,FamilyCandidate)] = []
    for (index,c) in candidates.enumerated() {
      guard let taken = date(c.taken) else { throw FamilyFailure.invalid }
      guard Int((day.timeIntervalSince1970-taken.timeIntervalSince1970)/86400) >= 335, !seen.contains(c.uuid) else { continue }
      let exact = String(c.taken.suffix(5)) == String(target.suffix(5))
      if c.source == "album" {
        ranked.append((exact ? 0 : 2, exact ? 0 : distance(taken,day), c.pixels, index, c))
      } else if c.source == "favourite" && exact && !albumIDs.contains(c.uuid) {
        ranked.append((1,0,c.pixels,index,c))
      }
    }
    ranked.sort { a,b in
      if a.0 != b.0 { return a.0 < b.0 }; if a.1 != b.1 { return a.1 < b.1 }
      if a.2 != b.2 { return a.2 > b.2 }; return a.3 < b.3
    }
    return (round, Array(ranked.prefix(12)).map { $0.4 })
  }
  static func readCandidates(_ file: String, people: [String:String], albumTitle: String = "Family Album") throws -> ([FamilyCandidate], Bool) {
    guard !albumTitle.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else { throw FamilyFailure.invalid }
    let db = try FamilyDatabase(file)
    let base = """
      SELECT a.ZUUID, a.ZDATECREATED, x.ZTIMEZONEOFFSET, a.ZWIDTH * a.ZHEIGHT
      FROM ZASSET a LEFT JOIN ZADDITIONALASSETATTRIBUTES x ON x.ZASSET = a.Z_PK
      WHERE a.ZKIND = 0 AND a.ZTRASHEDSTATE = 0 AND a.ZHIDDEN = 0
      AND COALESCE(a.ZKINDSUBTYPE, 0) != ? AND a.ZWIDTH > a.ZHEIGHT
      """
    let share = try db.query("SELECT Z_PK FROM ZSHARE WHERE ZTITLE = ?", [.text(albumTitle)]).first
    var out: [FamilyCandidate] = []
    func append(_ rows: [[FamilySQLValue]], _ source: String) throws {
      for r in rows {
        guard let uuid = r[0].text, let seconds = r[1].number else { throw FamilyFailure.blocked }
        let taken = Date(timeIntervalSince1970: 978307200 + seconds + (r[2].number ?? 0))
        out.append(FamilyCandidate(uuid:uuid, taken:iso(taken), source:source, pixels:r[3].integer ?? 0))
      }
    }
    if let pk = share?[0].integer {
      try append(db.query(base + " AND a.ZCOLLECTIONSHARE = ?", [.integer(10),.integer(pk)]), "album")
    }
    var pks: Set<Int64> = []
    // Python str.strip whitespace, including ASCII separators absent from Foundation's set.
    let pythonWhitespace = CharacterSet(charactersIn:"\u{9}\u{A}\u{B}\u{C}\u{D}\u{1C}\u{1D}\u{1E}\u{1F}\u{20}\u{85}\u{A0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}")
    let names = Set(people.values.flatMap { [$0, $0.trimmingCharacters(in:pythonWhitespace)] })
    for n in names {
      for r in try db.query("SELECT Z_PK FROM ZPERSON WHERE ZFULLNAME = ? OR ZDISPLAYNAME = ?", [.text(n),.text(n)]) {
        if let pk = r[0].integer { pks.insert(pk) }
      }
    }
    if !pks.isEmpty {
      let sorted = pks.sorted()
      let marks = Array(repeating:"?", count:sorted.count).joined(separator:",")
      let q = base + " AND a.ZFAVORITE = 1 AND EXISTS (SELECT 1 FROM ZDETECTEDFACE f WHERE f.ZASSETFORFACE = a.Z_PK AND f.ZHIDDEN = 0 AND f.ZPERSONFORFACE IN (\(marks)))"
      try append(db.query(q, [.integer(10)] + sorted.map { .integer($0) }), "favourite")
    }
    return (out, share != nil)
  }
}

private enum FamilySQLValue {
  case text(String), integer(Int64), real(Double), null
  var text: String? { if case .text(let s) = self { return s }; return nil }
  var number: Double? { switch self { case .integer(let n):return Double(n); case .real(let n):return n; default:return nil } }
  var integer: Int64? { if case .integer(let n) = self { return n }; return nil }
}
private final class FamilyDatabase {
  private var handle: OpaquePointer?
  init(_ file: String) throws {
    guard sqlite3_open_v2(file, &handle, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nil) == SQLITE_OK else {
      sqlite3_close(handle); handle = nil; throw FamilyFailure.blocked
    }
  }
  deinit { sqlite3_close(handle) }
  func query(_ sql: String, _ bindings: [FamilySQLValue]) throws -> [[FamilySQLValue]] {
    var statement: OpaquePointer?
    guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK else { throw FamilyFailure.blocked }
    defer { sqlite3_finalize(statement) }
    let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    for (index,b) in bindings.enumerated() {
      let rc: Int32
      switch b {
      case .integer(let n): rc = sqlite3_bind_int64(statement,Int32(index+1),n)
      case .text(let s): rc = sqlite3_bind_text(statement,Int32(index+1),s,-1,transient)
      default: throw FamilyFailure.blocked
      }
      guard rc == SQLITE_OK else { throw FamilyFailure.blocked }
    }
    var rows: [[FamilySQLValue]] = []
    while true {
      let rc = sqlite3_step(statement)
      if rc == SQLITE_DONE { return rows }
      guard rc == SQLITE_ROW else { throw FamilyFailure.blocked }
      rows.append((0..<sqlite3_column_count(statement)).map { i in
        switch sqlite3_column_type(statement,i) {
        case SQLITE_INTEGER: return .integer(sqlite3_column_int64(statement,i))
        case SQLITE_FLOAT: return .real(sqlite3_column_double(statement,i))
        case SQLITE_TEXT: return .text(String(cString:sqlite3_column_text(statement,i)))
        default:return .null
        }
      })
    }
  }
}
