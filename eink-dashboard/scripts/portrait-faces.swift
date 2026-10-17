// Finds the faces in upright images with Apple's Vision framework and prints them as JSON:
// one line per image, boxes in pixels with the origin at the top left, largest first.
// Used by prepare-portraits.py (DECISIONS #85). Run: swift scripts/portrait-faces.swift a.jpg b.jpg
import Foundation
import Vision
import ImageIO

for path in CommandLine.arguments.dropFirst() {
  let url = URL(fileURLWithPath: path)
  guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
        let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    print("{\"file\":\"\(path)\",\"error\":\"unreadable\"}")
    continue
  }
  let w = Double(img.width), h = Double(img.height)
  let req = VNDetectFaceRectanglesRequest()
  try? VNImageRequestHandler(cgImage: img, options: [:]).perform([req])
  let faces = (req.results ?? [])
    .map { f -> [String: Double] in
      let b = f.boundingBox
      return ["x": b.minX * w, "y": (1 - b.maxY) * h, "w": b.width * w, "h": b.height * h,
              "conf": Double(f.confidence)]
    }
    .sorted { $0["w"]! * $0["h"]! > $1["w"]! * $1["h"]! }
  let out: [String: Any] = ["file": path, "width": w, "height": h, "faces": faces]
  let data = try! JSONSerialization.data(withJSONObject: out)
  print(String(data: data, encoding: .utf8)!)
}
