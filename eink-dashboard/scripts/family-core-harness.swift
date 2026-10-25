// Synthetic-only harness. Not compiled into the app; no production override interface.
import Foundation

@main struct Harness {
  static func main() throws {
    let input = FileHandle.standardInput.readDataToEndOfFile()
    let o = try JSONSerialization.jsonObject(with: input) as! [String: Any]
    let op = o["op"] as! String
    var result: [String: Any] = [:]
    if op == "download" {
      let cs = try JSONDecoder().decode([FamilyCandidate].self, from:JSONSerialization.data(withJSONObject:o["candidates"]!))
      let widths = o["widths"] as! [Int]
      var attempts: [String] = []
      do {
        let winner = try FamilyCore.fetchWinner(cs) { c in
          let i = attempts.count; attempts.append(c.uuid)
          if widths[i] == -2 { throw FamilyFailure.blocked }
          if widths[i] == -1 { return nil }
          return (Data(), widths[i], 1000)
        }
        result = ["attempts":attempts,"winner":winner?.0.uuid ?? "none"]
      } catch { result = ["attempts":attempts,"blocked":true] }
    } else if op == "publish" {
      do {
        let receipt = try JSONDecoder().decode(FamilyReceipt.self, from: JSONSerialization.data(withJSONObject:o["receipt"]!))
        try FamilyFiles.publish(Data([255,216,255,217]), receipt:receipt, root:URL(fileURLWithPath:o["root"] as! String))
        result = ["ok":true]
      } catch { result = ["blocked":true] }
    } else if op == "denial" {
      let error = NSError(domain:o["domain"] as! String, code:o["code"] as! Int)
      result = ["denied":FamilyCore.protectedDenial(error)]
    } else if op == "snapshot" {
      do {
        result = try FamilyFiles.withSnapshot(library: URL(fileURLWithPath:o["library"] as! String), cache: URL(fileURLWithPath:o["cache"] as! String)) { db in
          if o["fail"] as? Bool == true { throw FamilyFailure.blocked }
          let (cs, _) = try FamilyCore.readCandidates(db.path, people: [:])
          return ["count": cs.count, "files": try FileManager.default.contentsOfDirectory(atPath:db.deletingLastPathComponent().path).sorted()]
        }
      } catch { result = ["blocked": true] }
    } else if op == "guard" {
      var reads = 0
      do {
        _ = try FamilyCore.authorized(o["authorized"] as! Bool) { reads += 1; return 1 }
        result = ["reads": reads, "blocked": false]
      } catch { result = ["reads": reads, "blocked": true] }
    } else if op == "request" {
      result = ["valid": FamilyCore.validRequest(o["date"] as! String, o["request"] as! String)]
    } else {
      let target = o["date"] as! String
      let shownData = try JSONSerialization.data(withJSONObject: o["shown"] ?? [])
      let shown = try JSONDecoder().decode([FamilyShown].self, from: shownData)
      let candidates: [FamilyCandidate]
      var found = true
      if op == "sql" {
        (candidates, found) = try FamilyCore.readCandidates(o["db"] as! String, people: o["people"] as! [String:String], albumTitle: o["albumTitle"] as? String ?? "Family Album")
      } else {
        candidates = try JSONDecoder().decode([FamilyCandidate].self,
          from: JSONSerialization.data(withJSONObject: o["candidates"]!))
      }
      let selection = try FamilyCore.choose(candidates, shown: shown, target: target)
      result = ["round": selection.0, "albumFound": found,
        "candidates": selection.1.map { ["uuid": $0.uuid, "taken": $0.taken,
          "source": $0.source, "pixels": $0.pixels, "localIdentifier": "\($0.uuid)/L0/001"] as [String:Any] }]
      if op == "sql" {
        result["readCandidates"] = candidates.map { ["uuid":$0.uuid,"taken":$0.taken,"source":$0.source,"pixels":$0.pixels] as [String:Any] }
      }
    }
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]))
  }
}
