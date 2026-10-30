#!/bin/zsh
cd -- "${0:A:h}" || exit 1
python3 - <<'PY'
import http.server
import functools
import webbrowser
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory='dist')
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
url = f'http://127.0.0.1:{server.server_port}'
print(f'Neural Atlas: {url}\nKapatmak için Ctrl+C.', flush=True)
webbrowser.open(url)
try:
    server.serve_forever()
except KeyboardInterrupt:
    server.server_close()
PY
