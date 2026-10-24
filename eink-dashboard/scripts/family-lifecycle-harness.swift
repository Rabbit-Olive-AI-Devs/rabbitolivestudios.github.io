// Offline protocol exerciser, never compiled into the application. No PhotoKit or private paths.
import Foundation
import Darwin
@main enum LifecycleHarness {
 static func main() {
  let a=CommandLine.arguments
  do {
   let lease=try FamilyOwnership(home:URL(fileURLWithPath:a[1]),day:a[2],request:a[3])
   do {
    try lease.check()
    func delay() throws {
     try Data("ready".utf8).write(to:URL(fileURLWithPath:a[4]))
     while !FileManager.default.fileExists(atPath:a[5]) { usleep(10000) }
     try lease.check()
    }
    if a.count>7 {
     let root=URL(fileURLWithPath:a[8]),out=URL(fileURLWithPath:a[7])
     try lease.validateOutput(out)
     try FamilyFiles.withSnapshot(library:root.appendingPathComponent("library"),cache:root.appendingPathComponent("snapshots")) { db in
      try lease.check()
      if a[9]=="snapshot-timeout" { try delay() }
      if a[9]=="cleanup-denied" {
       let sealed=db.deletingLastPathComponent().appendingPathComponent("sealed",isDirectory:true)
       try FileManager.default.createDirectory(at:sealed,withIntermediateDirectories:false)
       try Data("fixture".utf8).write(to:sealed.appendingPathComponent("fixture"))
       try FileManager.default.setAttributes([.posixPermissions:0],ofItemAtPath:sealed.path)
      }
      // Synthetic bytes copied by the production boundary, not a real Photos DB.
      guard try Data(contentsOf:db)==Data("synthetic only".utf8) else { throw FamilyFailure.blocked }
     }
     try lease.check()
     if a[9] != "snapshot-timeout" { try delay() } // Synthetic PhotoKit boundary.
     let receipt=FamilyReceipt(request:a[3],date:a[2],uuid:"22222222-2222-4222-8222-222222222222",taken:"2020-10-19",source:"album",round:1,width:1600,height:1000)
     let jpeg=Data([255,216,255,192,0,11,8,3,232,6,64,1,1,17,0,255,217])
     try lease.check();try FamilyFiles.publish(jpeg,receipt:receipt,root:out)
    } else { try delay() }
    try lease.check()
    try Data("protected".utf8).write(to:URL(fileURLWithPath:a[6]))
    try lease.finish(status:"complete")
   } catch {
    if case FamilyFailure.cleanupIncomplete = error { exit(4) }
    try lease.finish(status:"cancelled")
   }
   withExtendedLifetime(lease) {}
  } catch { exit(3) }
 }
}
