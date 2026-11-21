#!/usr/bin/env python3
"""Ham kayit sunucusu — MCP arac girdilerini ve ham yanitlarini bayt-bayt diske yazar.

GET  /<ad>          -> girdi dosyasini oldugu gibi verir (MCP araci metni buradan alir;
                       elle yeniden yazim yok => rakam kaymasi riski yok)
POST /save?name=x   -> govdeyi ham JSON kaydi olarak <OUT>/x dosyasina yazar

Kosum: python3 kayit_sunucusu.py <port>   (bu dizinden; OUT = bu dizin)
"""
from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

OUT = Path(__file__).resolve().parent
GIRI = Path("/Users/serkankose/Documents/quann-3d/docs/patent/pct_v2/filing")
İZIN = {
    "claims": GIRI / "CLAIMS_FILING-COPY_EN.txt",
    "description": GIRI / "DESCRIPTION_FILING-COPY_EN.txt",
    "abstract": GIRI / "ABSTRACT_FILING-COPY_EN.txt",
    "tr_claims": GIRI / "ISTEMLER_FILING-COPY_TR.txt",
    "tr_description": GIRI / "TARIFNAME_FILING-COPY_TR.txt",
    "tr_abstract": GIRI / "OZET_FILING-COPY_TR.txt",
}


class H(BaseHTTPRequestHandler):
    def _gönder(self, kod: int, veri: bytes, tip: str) -> None:
        self.send_response(kod)
        self.send_header("Content-Type", tip)
        self.send_header("Content-Length", str(len(veri)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(veri)

    def do_GET(self) -> None:  # noqa: N802
        ad = urlparse(self.path).path.strip("/")
        hedef = İZIN.get(ad)
        if hedef is None or not hedef.is_file():
            self._gönder(404, b"yok", "text/plain; charset=utf-8")
            return
        self._gönder(200, hedef.read_bytes(), "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        q = parse_qs(urlparse(self.path).query)
        ad = (q.get("name") or ["kayit.json"])[0]
        if "/" in ad or ".." in ad or not ad.endswith(".json"):
            self._gönder(400, b"ad gecersiz", "text/plain; charset=utf-8")
            return
        n = int(self.headers.get("Content-Length", "0"))
        govde = self.rfile.read(n)
        (OUT / ad).write_bytes(govde)
        self._gönder(200, f"yazildi {ad} {len(govde)} B\n".encode(), "text/plain; charset=utf-8")

    def log_message(self, *a) -> None:  # suskun
        return


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
    print(f"dinliyor {port} · girdiler: {', '.join(İZIN)} · kayit: {OUT}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
