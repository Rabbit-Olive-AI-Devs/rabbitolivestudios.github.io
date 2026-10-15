"""Pull the picture out of a built series page, to look at it: python3 -E -P page-png.py <page.html> <out.png>"""
import base64
import re
import sys

html = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
if not m:
    sys.exit("no picture in " + sys.argv[1])
open(sys.argv[2], "wb").write(base64.b64decode(m.group(1)))
