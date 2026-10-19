"""Inline the dashboard's script and images so one HTML file opens anywhere.

Usage: python -I scripts/bundle_dashboard.py <index.html> <assets dir> <out.html>
"""

import base64
import re
import sys
from pathlib import Path

src, assets, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
s = src.read_text()
js = (assets / "plotly.min.js").read_text()
s = s.replace(
    '<script src="assets/plotly.min.js"></script>',
    "<script>" + js.replace("</script>", "<\\/script>") + "</script>",
)


def img(m: re.Match) -> str:
    data = base64.b64encode((assets / m.group(1)).read_bytes()).decode()
    return f'src="data:image/png;base64,{data}"'


s = re.sub(r'src="assets/([^"]+\.png)"', img, s)
assert "assets/" not in s, "unresolved asset"
out.write_text(s)
print(out, f"{out.stat().st_size / 1e6:.1f} MB")
