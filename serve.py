"""Serves the front end at http://localhost:8000  (python serve.py)."""
import http.server, os, functools, webbrowser
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
print("Open http://localhost:8000  (Ctrl+C to stop)")
webbrowser.open("http://localhost:8000")
http.server.ThreadingHTTPServer(("", 8000), h).serve_forever()
