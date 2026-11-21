r"""UYGUNLUK_NOTU.md §1 kural matrisini yeniden kurar — her künye yazim aninda dogrulanir.

Neden: 2026-11-19 12:09'da kurulan genisletilmis matris (22 satir) 12:13 sonrasi dosyada
bulunamadi (nedeni dogrulanmadi; tek kanit: 12:12 yedeginde 9 satirlik eski matris var).
Bu betik matrisi **ureterek** yazar: her `dosya:satir = DEGER` kumesi yazimdan hemen once
ham kaynakta kontrol edilir; biri bile tutmazsa betik HICBIR SEY YAZMAZ.

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/yeniden_bolum1.py
Cikis: 0 = §1 yazildi ve her künye acildi; 1 = künye dogrulanamadi (dosya degismez).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
NOT = REPO / "docs/patent/pct_v2/UYGUNLUK_NOTU.md"
CIKTI = "patent-cikti/pct/pct_v2-pct-inceleme"
MOTOR = "~/patent-creator-venv/lib/python3.13/site-packages/mcp_server"
A, B = f"{CIKTI}/A_pct_en.json", f"{CIKTI}/B_pct_tr.json"
URETIM = f"{CIKTI}/uretim.json"
G = f"{CIKTI}/G_epo_formal_en.json"
D, E, F = f"{CIKTI}/D_epo_claims_en.json", f"{CIKTI}/E_epo_claims_tr.json", f"{CIKTI}/F_epo_spec_support_en.json"
KANIT = f"{CIKTI}/KANIT_antecedent.json"
LC, LT = f"{CIKTI}/lint_CLAIMS_EN_md.txt", f"{CIKTI}/lint_ISTEMLER_TR_md.txt"
CE, CT = "docs/patent/pct_v2/CLAIMS_EN.md", "docs/patent/pct_v2/ISTEMLER_TR.md"
DE, DT = "docs/patent/pct_v2/DESCRIPTION_EN.md", "docs/patent/pct_v2/DESCRIPTION_TR.md"
OZ, OLC = "docs/patent/pct_v2/OZET_PCT.md", "docs/patent/pct_v2/OLCUM_KAYITLARI.md"
DEN = "patent-cikti/pct/pct_v2-denetmen/DENETIM_OZET.md"
PCT = f"{MOTOR}/pct_formalities_checker.py"

# --- kural matrisi: (Kural, istenen, paketin durumu, kanit hucre parcalari) ---------
# kanit parcasinin bicimi:  "dosya:satir = DEGER"  -> DEGER o satirda birebir gecmeli
#                           "dosya:satir"          -> satir acilmali ve bos olmamali
#                           "dosya:satir-satir"    -> aralikta en az bir dolu satir
#                           duz metin              -> oldigi gibi yazilir, denetim yok
SATIRLAR = [
    ("PCT Rule 4.3", "Başlık kısa ve teknik alanla uyumlu",
     "EN altı · TR yedi kelime; iki dilde de `compliant: true`",
     [f"`{A}:12 = 6`", f"`{A}:13 = true`", f"`{B}:12 = 7`", f"`{B}:13 = true`"]),

    ("PCT Rule 5.1(a)", "Teknik alan · önceki teknik · buluşun açıklanması · şekillerin kısa açıklanması · "
     "en iyi biçim (+ sanayide uygulanabilirlik)",
     "Altı başlık iki dilde de sırayla; motorun denetlediği beş alanın eksik listesi boş",
     [f"`{A}:24 = []`", f"`{A}:25 = true`", f"`{B}:24 = []`", f"`{B}:25 = true`",
      f"`{DE}:11,21,39,67,87,217` (EN başlıkları)", f"`{DT}:11,21,39,67,87,217` (TR başlıkları)"]),

    ("PCT Rule 6.1(b)", "İstemler ardışık numaralı",
     "On beş istem; numaralar 1–15 kesintisiz",
     [f"`{A}:28 = 15`", f"`{A}:29-45` (`claim_numbers` 1…15)", f"`{B}:28 = 15`",
      f"`{URETIM}:40 = 15`", f"`{URETIM}:82 = 15`"]),

    ("PCT Rule 6.2(a)", "İstem gövdesinde tarifname/şekil atfı yok",
     "EN gövdesinde tarama boş; TR'deki tek eşleşme \"hesaplayacak **şekilde**\" sözcüğü — şekil atfı değil",
     [f"`{CE}:10-66` gövde taraması (`grep -c \"FIG\\.\"` = 0)",
      f"`{CT}:11-39` taraması; eşleşme `{CT}:37`"]),

    ("PCT Rule 6.2(b)", "Referans işaretleri, anlaşmayı hızlandırmıyorsa zorunlu değil",
     "Bilinçli tasarım kararı: hiç işaret yok (numara şeması uydurulmadı); kural bu eksikliği uyarıya çeviriyor",
     [f"`{CE}:7` tasarım notu", f"`{G}:11 = false` (`has_reference_signs`)",
      f"`{G}:9` (\"Missing reference signs\")", f"`{G}:73 = true` (`overall_compliant`)", "→ §4.4"]),

    ("PCT Rule 5.2(a) / 11.13(m)", "Paragraf numaralandırması",
     "İki tarifnede de yetmiş paragraf, aralıksız",
     [f"`{URETIM}:14 = 70`", f"`{URETIM}:15 = true`", f"`{URETIM}:56 = 70`", f"`{URETIM}:57 = true`"]),

    ("PCT Rule 8.1(b)", "Özet elli–yüz elli kelime",
     "EN yüz otuz yedi · TR yüz altı kelime; iki dilde `compliant: true`",
     [f"`{A}:6 = 137`", f"`{A}:7 = true`", f"`{B}:6 = 106`", f"`{B}:7 = true`",
      f"`{OZ}:48-49` (`wc -w` ölçümü)"]),

    ("PCT Rule 9.1 (birlik)", "Tek genel konsept; çokluk bağımlı istemle toplanır",
     "Üç bağımsız (yöntem 1 · sistem 13 · kayıt ortamı 15) + on iki bağımlı; yerel lint `claim_unity` **info**",
     [f"`{LC}:26 = info`", f"`{LT}:11 = info`", f"`{CE}:10` `{CE}:52` `{CE}:66`",
      f"`{CT}:11` `{CT}:35` `{CT}:39`"]),

    ("PCT Rule 11.13(l)–(n)", "Metin temizliği: birim/terim/sıcaklık tutarlılığı",
     "Terim denetimi iki dilde de temiz: metrik dışı birim ve aykırı sıcaklık listeleri boş",
     [f"`{A}:50 = []`", f"`{A}:51 = []`", f"`{A}:52 = true`", f"`{B}:50 = []`", f"`{B}:51 = []`",
      f"`{B}:52 = true`"]),

    ("AGDD §205 / Kural 5.2(a)", "Şekillerin kısa açıklanması gerçek şekil listesine bağlı",
     "Şekil 1–8 için ayrı paragraf; ölçüm taşıyanlar Şekil 6–7 olarak ayrıca işaretli",
     [f"`{DE}:83` (FIG. 8 tarif) ", f"`{DE}:85` (ölçüm taşıyan şekiller)", f"`{DT}:83`", f"`{DT}:85`"]),

    ("AGDD 5.2(b) — özetle okunacak şekil", "Özetle birlikte yayınlanacak şekil talep formunda işaretlenir",
     "**eksik/bekleniyor** — özet metninde şekil atfı yok; motor bu kontrolü çalıştırmıyor, "
     "işaret PCT/RO/101'de verilecek",
     [f"`{A}:9 = not checked`", f"`{A}:106-107` (atlanan kontrol + gerekçe)", f"`{PCT}:238-241` (motor kaynağı)",
      f"`{DEN}:59` (kapsam dışı listesi)"]),

    ("AGDD 5.2(b) — fiziksel vurgu", "Kalın/altı çizili/tasarım tarifi kuralları",
     "**eksik/bekleniyor** — başvuru metinlerinde (`.txt` nüshalar ve EN/TR kaynaklarında kalın-italik"
     "ayrımı yok) kalın ve altı çizili yok; fiziksel çıktı vekilde basılacak",
     ["`grep -c '\\*\\*\\|__' docs/patent/pct_v2/filing/*.txt` = 0/0/0/0/0/0 (bu oturumda koştu, "
      "başvuru metinleri)",
      f"`{A}:102-103` (`physical` atlandı)", f"`{PCT}:231-232` (motor kaynağı)"]),

    ("PCT Rule 13.1", "Şekiller ayrı sayfalarda, `FIG.` etiketli; şekil listesi tarifnamede",
     "**kısmen / şekil dosyaları denetlenmedi** — tarifnamedeki Şekil 1–8 listesi ve `FIG.` etiketleri yerinde; "
     "`fig/*.svg` içerik denetimi yapılmadı",
     [f"`{DE}:67` (başlık) ", f"`{DE}:85`", f"`{A}:9 = not checked`", f"`{DEN}:59`"]),

    ("PCT Rule 11.2–11.16", "Fiziksel biçim: A4, marj, satır numarası, punto, sayfa üstü",
     "**eksik/bekleniyor** — motor metinden doğrulayamığını açıkça söylüyor; fiziksel kopya vekil tarafında üretilecek",
     [f"`{A}:102-103` (`physical` + gerekçe)", f"`{A}:110` (`verdict_scope`)", f"`{PCT}:231-232`",
      f"`{PCT}:252-255` (\"This is not a filing clearance.\")"]),

    ("PCT Rule 19.2", "Değiştirilmiş sayfa gönderimi (istemin tarife göre düzeltilmesi)",
     "**bekleniyor** — uluslararası arama raporundan sonra; bu oturumda gönderilecek bir değişiklik sayfası yok",
     [f"`{A}:110` (`verdict_scope`)", f"`{DEN}:57-60` (denetlenemedi listesi)"]),

    ("PCT/RO/101 talep formu", "Başvuru formu: mucitler, vekâletname, beyanlar, şekil atfı",
     "**eksik** — paket metinlerden oluşuyor, form hazırlanmadı",
     [f"`{A}:110` (\"forms … require manual review\")", f"`{DEN}:59`"]),

    ("PCT Rule 6.4(b) / 3.2 — ücret ve vekâletname", "İade ücreti ve vekâletname beyanı",
     "**eksik** — ücret hesabı ve vekâletname vekil/ofis işi; bu paket ücret iddiası taşımıyor",
     [f"`{A}:110` (\"fee payment require manual review\")", f"`{DEN}:59`"]),

    ("EPO UygK 45(2) / 46(1)", "İstemler tarifnameyle destekli; tarifname buluşu açıklar",
     "On beş istemin on beşi destekli, kapsama yüzde yüz, karşılanmayan istem yok",
     [f"`{F}:629 = true`", f"`{F}:631 = 100`", f"`{F}:632 = 15`", f"`{F}:634 = []`"]),

    ("EPO UygK 43(1)–(2) · İBK F-II 3.7", "İstem açık ve öz; kategori açık",
     "EN'de on beş `Rule 43(1)` eşleşmesinin on beşi de İngilizce-belirteç artefaktı; "
     "TR istem linti **0 hata**; gerçek kusur sayacı sıfır",
     [f"`{D}:4 = 1`", f"`{D}:5 = 0`", f"`{E}:4 = 16`", f"`{E}:5 = 0`",
      f"`{KANIT}:10 = 0` (`D_GECEK_KUSUR`, EN)", f"`{KANIT}:113 = 0` (TR)", "→ §4.1–§4.2"]),

    ("İBK F-IV 5.1", "Bağımsız istem iki parçalı: ön izleme + ayırt edici bölüm",
     "Üç bağımsız istemin her birinde tam bir `characterized in that` / `karakterize edilir`; "
     "bağımlı kuyrukları `tanımlıdır.`",
     [f"`{CE}:10` `{CE}:52` `{CE}:66` (grep toplamı 3)", f"`{CT}:11` `{CT}:35` `{CT}:39` (grep toplamı 3)",
      f"`{CT}:13` (örnek bağımlı kuyruk; grep toplamı 12)"]),

    ("İBK H-V 1", "Teknik sorun + teknik etki ölçülü biçimde belirtilmiş",
     "Etki yalnız ölçülen eksenler üzerinde ve ölçüm koşuluyla yazıldı; doğruluk üstünlüğü iddiası yok",
     [f"`{DE}:123` (k=13→50 maliyet sınırı)", f"`{DT}:123`", f"`{OLC}:51-52` (oranların ham künyesi)"]),

    ("İki dillilik", "TR metin EN aslının birebir karşılığı; terimler tek biçimli",
     "Yetmiş–yetmiş paragraf, on beş–on beş istem; bölüm başlıkları iki dilde paralel — "
     "**birebir çeviri denetimi yapılmadı**, yalnız yapı + ana terim listesi karşılaştırıldı",
     [f"`{URETIM}:14-15`", f"`{URETIM}:56-57`", f"`{DT}:11,21,39,67,87,217`", "→ §5"]),
]

BASLIK = """## 1. Kural matrisi

Her satır bir kural maddesi; **durum** sütununda `eksik`/`bekleniyor` yazan madde bu paketin
karşılamadığı veya karşılayamayacağı işdir ve §5'te karşılığı vardır. Kanıt hücresinde
`dosya:satır = DEĞER` biçimindeki her künye yazım anında ham kaynakta doğrulandı; aynı
adresları her koşumda yeniden açan kapı `patent-cikti/pct/pct_v2-pct-inceleme/dogrul_tablolar.py`
(§2'deki motor değerlerini de yeniden okur).

| Kural | İstenen | Paketin durumu | Kanıt (bu oturumda okundu) |
|---|---|---|---|
"""


def coz(yol: str) -> Path | None:
    p = Path(yol).expanduser()
    return p if p.exists() else (REPO / yol if (REPO / yol).exists() else None)


ADRES = re.compile(r"([^`\s=]+?):([0-9,\-]+)")


def dogrula(kanitler: list[str]) -> list[str]:
    """her `dosya:satir[ = DEGER]` kumesini ham kaynakta acar; tutmayanlari dondurur."""
    hata: list[str] = []
    for kanit in kanitler:
        for yol, spec in ADRES.findall(kanit):
            p = coz(yol)
            if p is None:
                hata.append(f"DOSYA YOK {yol}")
                continue
            satirlar = p.read_text(encoding="utf-8", errors="replace").splitlines()
            for spec_no in spec.split(","):
                if "-" in spec_no:
                    # aralik: sinirlar dosya icinde olmali ve aralik hicbir satiri bos OLMAYAN
                    # bir icerik tasimali (markdown'da paragraflar arasindaki bos satirlar normaldir)
                    a, b = (int(x) for x in spec_no.split("-", 1))
                    if not (1 <= a <= b <= len(satirlar)):
                        hata.append(f"ARALIK SINIRI DIŞINDA {yol}:{spec_no} (dosya {len(satirlar)} satır)")
                    elif not any(satirlar[n - 1].strip() for n in range(a, b + 1)):
                        hata.append(f"ARALIK BOŞ {yol}:{spec_no}")
                else:
                    n = int(spec_no)
                    if not (1 <= n <= len(satirlar)) or not satirlar[n - 1].strip():
                        hata.append(f"SATIR ACILMIYOR {yol}:{spec_no}")
        # " = DEGER" istenirse DEGER o satirda birebir gecmeli
        for yol, n, deger in re.findall(r"([^`\s=]+?):(\d+) = (\[\]|true|false|[0-9eE.+-]+|[A-Za-z ]{3,12})",
                                        kanit):
            p = coz(yol)
            if p is None:
                hata.append(f"DOSYA YOK {yol}")
                continue
            s = p.read_text(encoding="utf-8", errors="replace").splitlines()
            if not (1 <= int(n) <= len(s)) or deger not in s[int(n) - 1]:
                hata.append(f"DEGER TUTMUYOR {yol}:{n} != {deger!r} "
                            f"(satır: {s[int(n) - 1][:60]!r})")
    return hata


def _isaret(durum: str) -> bool:
    """durum sutununda ACIK kural durumu olarak `eksik`/`bekleniyor` yazili mi?
    (duz metin icindeki 'eksik listesi' gibi sozcukler sayilmaz)"""
    return bool(re.search(r"\*\*(eksik|bekleniyor)", durum))


def main() -> int:
    govde = [BASLIK]
    hatalar: list[str] = []
    for kural, istenen, durum, kanitler in SATIRLAR:
        hatalar += [f"[{kural}] {h}" for h in dogrula(kanitler)]
        govde.append(f"| {kural} | {istenen} | {durum} | {' · '.join(kanitler)} |\n")
    if hatalar:
        print(f"BULGU: {len(hatalar)} künye doğrulanamadı — dosyaya DOKUNULMADI")
        for h in hatalar:
            print("  BULGU  " + h)
        return 1

    n_eksik = sum(1 for *_x, d, _k in [(k, i, d, kk) for k, i, d, kk in SATIRLAR] if _isaret(d))
    n_kismen = sum(1 for _k, _i, d, _kk in SATIRLAR if "kısmen" in d)
    govde.append(f"\nToplam {len(SATIRLAR)} satır · {n_eksik} satırda durum açıkça "
                 f"`eksik`/`bekleniyor` · {n_kismen} satır `kısmen` — bu üç sayı "
                 "`patent-cikti/pct/pct_v2-pct-inceleme/yeniden_bolum1.py:39-147` "
                 "`SATIRLAR` listesindeki durum metinlerinden üretilir.\n")
    metin = NOT.read_text(encoding="utf-8")
    bas = metin.index("## 1. Kural matrisi")
    son = metin.index("## 2. Motor bataryası")
    YED = REPO / "patent-cikti/pct/pct_v2-pct-inceleme/oncesi/UYGUNLUK_NOTU.md.bolum1-oncesi"
    YED.parent.mkdir(parents=True, exist_ok=True)
    YED.write_text(metin, encoding="utf-8")
    NOT.write_text(metin[:bas] + "".join(govde) + "\n" + metin[son:], encoding="utf-8")
    t = NOT.read_text(encoding="utf-8")
    print(f"OK: §1 yeniden kuruldu — {len(SATIRLAR)} satır · {n_eksik} satır eksik/bekleniyor "
          f"· {n_kismen} satır kısmen")
    print(f"-> {NOT.relative_to(REPO)} ({len(t.splitlines())} satır) · yedek {YED.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
