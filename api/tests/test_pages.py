#!/usr/bin/env python3
"""Every page must be served with and without a trailing slash, and a missing file must be a 404.

    test_pages.py

Runs the app in a TestClient over a web app directory made for the test, so it needs no database:
the one call the page route makes to PostgreSQL is replaced.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starlette.routing import Match  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import text_pair  # noqa: E402

FAILURES = []

PAGES = ("", "/search", "/time", "/group/12", "/sorted-results", "/text-view", "/network", "/semantic")


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def make_app_dir(root):
    dist = os.path.join(root, "testdb", "dist")
    os.makedirs(os.path.join(dist, "assets"))
    with open(os.path.join(root, "testdb", "appConfig.json"), "w", encoding="utf8") as config:
        config.write('{"databaseName": "testdb", "showNetworkGraph": true}')
    files = {
        "index.html": b"<html><head></head><body></body></html>",
        "favicon.ico": b"\x00\x00\x01\x00",
        "assets/index-abc.js": "console.log('é')".encode(),
        "assets/index-abc.css": b"body{}",
        "assets/font-abc.woff2": b"wOF2\x00\xff\xfe",
    }
    for name, content in files.items():
        with open(os.path.join(dist, name), "wb") as output:
            output.write(content)


def endpoint(path, method="GET"):
    """The name of the function the app routes a request to."""
    scope = {"type": "http", "path": path, "method": method, "root_path": ""}
    for route in text_pair.app.router.routes:
        if route.matches(scope)[0] == Match.FULL:
            return route.endpoint.__name__
    return None


def main():
    root = tempfile.mkdtemp()
    make_app_dir(root)
    text_pair.APP_PATH = root
    text_pair.has_passage_families = lambda table: False
    client = TestClient(text_pair.app, raise_server_exceptions=False)  # a failing route is a 500, as served

    for prefix in ("", "/text-pair"):
        for page in PAGES:
            for path in (f"{prefix}/testdb{page}", f"{prefix}/testdb{page}/"):
                response = client.get(path, follow_redirects=False)
                check(f"{path} is the page, not a redirect",
                      response.status_code == 200 and "window.textpairConfig" in response.text)
    for path in ("/", "/text-pair", "/text-pair/"):
        response = client.get(path, follow_redirects=False)
        check(f"{path} lists the databases", response.status_code == 200 and "testdb" in response.text)
    check("an unknown database is a 404", client.get("/nodb/", follow_redirects=False).status_code == 404)

    # A route added with a trailing slash must not take an API route's requests
    for path, name in (("/database_stats/", "database_stats"), ("/text-pair-api/database_stats/", "database_stats"),
                       ("/metadata/", "metadata"), ("/group/12/", "get_passage_group"),
                       ("/sorted_results/", "get_sorted_results"), ("/testdb/retrieve_all_docs/", "retrieve_all")):
        check(f"{path} still goes to {name}", endpoint(path) == name)

    for prefix in ("", "/text-pair"):
        response = client.get(f"{prefix}/testdb/assets/index-abc.js")
        check(f"{prefix}/testdb/assets: a script is served as JavaScript",
              response.status_code == 200 and response.headers["content-type"].startswith("application/javascript")
              and response.content == "console.log('é')".encode())
    response = client.get("/testdb/assets/index-abc.css")
    check("a style sheet is served as CSS",
          response.status_code == 200 and response.headers["content-type"].startswith("text/css"))
    response = client.get("/testdb/assets/font-abc.woff2")
    check("another asset is served as it is, typed by its extension",
          response.status_code == 200 and response.headers["content-type"] == "font/woff2"
          and response.content == b"wOF2\x00\xff\xfe")
    response = client.get("/testdb/favicon.ico")
    check("the favicon is served", response.status_code == 200 and response.content == b"\x00\x00\x01\x00")

    for path in ("/testdb/assets/index-old.js", "/text-pair/testdb/assets/index-old.css", "/testdb/css/old.css",
                 "/testdb/js/old.js", "/nodb/assets/index-abc.js", "/testdb/assets/%2e%2e"):
        check(f"{path} is a 404", client.get(path).status_code == 404)
    os.remove(os.path.join(root, "testdb", "dist", "favicon.ico"))
    check("a missing favicon is a 404", client.get("/testdb/favicon.ico").status_code == 404)

    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
