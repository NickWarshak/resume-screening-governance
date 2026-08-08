"""
Gate: docs/demo.html must actually render and populate in a real browser.

verify_web_demo.py proves the scoring arithmetic is right. This proves the page
is wired up: controls built, scores rendered, attribution chart populated, the
decision envelope resolved, and the governed model showing exactly zero
sensitivity to every proxy feature.

Serves docs/ over HTTP, loads the page in headless Chrome or Edge, and parses
the dumped DOM into a tree so assertions do not depend on attribute
serialization order. Skips (exit 0) if no Chromium browser is installed.

Run:  python verify_demo_render.py
"""

import functools
import http.server
import os
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
from html.parser import HTMLParser

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOCS = os.path.join(BASE, "docs")

BROWSERS = [
    os.path.expandvars(p) for p in (
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
        r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
]
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class Node:
    def __init__(self, tag, attrs=None):
        self.tag, self.attrs, self.kids, self.text = tag, dict(attrs or {}), [], ""

    @property
    def cls(self):
        return (self.attrs.get("class") or "").split()

    def walk(self):
        yield self
        for k in self.kids:
            yield from k.walk()

    def find_id(self, i):
        return next((n for n in self.walk() if n.attrs.get("id") == i), None)

    def by_class(self, c):
        return [n for n in self.walk() if c in n.cls]

    def all_text(self):
        return (self.text + "".join(k.all_text() for k in self.kids)).strip()


class Tree(HTMLParser):
    """Minimal DOM builder. Ignores <script>/<style> content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
            return
        n = Node(tag, attrs)
        self.stack[-1].kids.append(n)
        if tag not in VOID:
            self.stack.append(n)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if not self.skip:
            self.stack[-1].text += data


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def render(browser):
    handler = functools.partial(QuietHandler, directory=DOCS)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as prof:
            res = subprocess.run(
                [browser, "--headless", "--disable-gpu", "--no-sandbox",
                 "--virtual-time-budget=8000", f"--user-data-dir={prof}", "--dump-dom",
                 f"http://127.0.0.1:{port}/demo.html"],
                capture_output=True, timeout=180,
            )
    finally:
        httpd.shutdown()
    return res.stdout.decode("utf-8", errors="replace")


def main():
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        print("SKIP: no Chromium-based browser found; cannot render-test the demo.")
        return 0
    print(f"rendering docs/demo.html in {os.path.basename(browser)}")

    dom = render(browser)
    print(f"dumped DOM: {len(dom):,} chars")
    if len(dom) < 2000:
        print("FAIL: browser produced no usable DOM")
        return 1

    t = Tree()
    t.feed(dom)
    root = t.root
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))
        if not cond:
            fails.append(name)

    app = root.find_id("app")
    check("loader hidden", "hidden" in root.find_id("loading").attrs)
    check("app revealed", app is not None and "hidden" not in app.attrs)

    for pid, label in (("gov-num", "governed"), ("nai-num", "naive")):
        txt = root.find_id(pid).all_text()
        check(f"{label} score rendered", bool(re.match(r"^[\d.]+%", txt)), txt[:16])

    check("16 controls built", len(root.by_class("ctl")) == 16, str(len(root.by_class("ctl"))))
    for f in ("years_experience", "gpa", "education_level", "referral",
              "top_school", "employment_gap_months", "distance_from_office_km"):
        check(f"control {f}", root.find_id("in-" + f) is not None)

    crows = root.find_id("contrib").by_class("crow")
    check("12 attribution rows", len(crows) == 12, str(len(crows)))
    check("attribution bars sized", all(
        "width" in (r.by_class("cfill")[0].attrs.get("style") or "") for r in crows))

    lev = root.find_id("leverage")
    check("4 proxy leverage rows", len([n for n in lev.walk() if n.tag == "tr"]) == 4)
    zeros = [n.all_text() for n in lev.by_class("zero")]
    check("governed model has 0.0 sensitivity to every proxy",
          len(zeros) == 4 and all(z == "0.0 pts" for z in zeros), str(zeros))
    hot = [n.all_text() for n in lev.by_class("hot")]
    check("naive model is proxy-sensitive",
          sum(1 for s in hot if not re.match(r"^[+-]?0\.0 ", s)) >= 3, str(hot))

    act = root.find_id("gov-action")
    check("decision envelope resolved",
          any(c in act.cls for c in ("advance", "abstain", "low")), " ".join(act.cls))
    check("4 preset buttons", len(root.by_class("btn")) == 4)
    check("population percentile shown", "test set" in root.find_id("gov-sub").all_text())
    check("no unreplaced placeholder", "—" not in root.find_id("gov-num").all_text())

    print()
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        return 1
    print("All render checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
