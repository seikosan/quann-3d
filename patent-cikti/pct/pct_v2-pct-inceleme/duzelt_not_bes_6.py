r"""UYGUNLUK_NOTU §5 ve §6'nın son bloğunu + OLCUM künyesini kanıt sütununa bağlar.

Neden: §5 ve §6'daki sayılar ya kural kimliği (PCT Rule 11.2–11.16 gibi) ya da
"kosulmadi" ifadeleriydi ve satirda hic atif yoktu. Her satir artik kendi ham
kaynagini tasiyor: motorun `checks_skipped` kaydi, J'nin opt-in kapisi, arayüz
aracinin tanim oldugu dosya. Içermik birebir korunuyor, yalnizca izlenebilirlik ekleniyor.

Kosum: python3 patent-cikti/pct/pct_v2-pct-inceleme/duzelt_not_bes_6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
NOT = REPO / "docs/patent/pct_v2/UYGUNLUK_NOTU.md"
DEFTER = REPO / "docs/patent/pct_v2/OLCUM_KAYITLARI.md"
PC = "patent-cikti/pct/pct_v2-pct-inceleme"
VENV = "~/patent-creator-venv/lib/python3.13/site-packages/mcp_server"

BES = f"""Her satır motorun kendi kaydına bağlıdır; bu bölümde hiçbir kontrol "temiz" sayılmaz.

| Koşulmayan kontrol | Neden koşulmadı | Kanıt satırı |
|---|---|---|
| Fiziksel biçim (PCT Rule 11.2–11.16: A4, kenar boşluğu, satır arası, yazı tipi, sayfa numarası) | Motor düz metinden fiziksel gereklilikleri denetleyemediğini kendisi bildiriyor: "physical requirements cannot be verified from text; manual inspection required". PDF dizgisi olmadığı için denetlenmedi. | `{PC}/A_pct_en.json:102-103` (`"check": "physical"` + gerekçe) |
| Özette şekil tarifi (Rule 8.1(b)) | Özetle hangi şeklin yayımlanacağı başvuru formunda seçilir; metinden bilinemez: "the figure to publish with the abstract is chosen on the request form". | `{PC}/A_pct_en.json:105-107` (`"check": "abstract figure designation"`) |
| Yenilik / ön sanat araması — **YAPILMADI** | Bu oturumda hiçbir veritabanı araması koşulmadı; arayüz aracının hazır olup olmadığını bildiren hazırlık ölçümü de çağrılmadı. Rule 33 (ISRS) anlamında bir değerlendirme yoktur. | arama aracı tanımı: `{VENV}/tools/bigquery_tools.py:96-97`; bu depoda arama kaydı yok |
| Şekil dosyalarının içeriği (Rule 11.12–11.14 çizgi/kalite) | Motorun `drawings_present` beyanı çalıştırıldı ama kendi kapsam satırı bunu "a declaration, not inspection of drawing files" olarak işaretliyor. Çizim dosyası denetimi yapılmadı. | `{PC}/A_pct_en.json:110` (`verdict_scope`) |
| US öncül dayanak denetimi | Motor bu denetimi opt-in kapalı tutuyor; `PATENT_ENABLE_ANTECEDENT_CHECK=1` hiç koşulmadı. Yerine yerel hat koşusu ve §4.5 sınıflandırması var. | `{PC}/J_us_claims_en.json:17-19` (kural + `enable_with`) |
| İdari katman | Form PCT/RO/101, ücret ve indirim hesabı, vekaletname, RO TÜRKPATENT başvuru dili tercihi, Rule 26bis beyanı, dizim listesi (ST.26) kapsam dışı. | idari veri bu depoda üretilmiyor; §8 vekil onayı satırı |
| Bağımsız onay | Bu notun bulguları ana oturumdan; vekil ve bağımsız doğrulayıcı turu olmadan başvuru kararı verilmez. | §8 karar satırı |
"""

VEKIL = f"""**Vekile bırakılanlar (metin değişikliği yapılmadı):**

| Madde | Neden burada değiştirilmedi | Kanıt satırı |
|---|---|---|
| TR bağımsız istemin tek cümle kuralına takılan çok sıralı yapısı | EN'de de aynı uzunlukta; kapsam daraltmadan tek cümleye indirmek bağımsız istemin korunmasını riske atıyor | `docs/patent/pct_v2/ISTEMLER_TR.md:35` ↔ `docs/patent/pct_v2/CLAIMS_EN.md:52` |
| Çizim referans numaraları şeması | Sayı uydurmamak için konmadı (§1, Rule 6.2(b) satırı); tarifname şekilleri `FIG. N` olarak anıyor | `docs/patent/pct_v2/DESCRIPTION_EN.md:105` (FIG. 3 anışı) |
| TR [0057]'deki "vekil (proxy)" niteleyicisinin kapsam iddiasına etkisi | Niteleyici ham örnekemin kendi adlandırması; metne taşındı ama kapsam yorumu hukuki karar | `fraud/logs/run_full_scale_edge_eval.log:46` → `docs/patent/pct_v2/DESCRIPTION_TR.md:181` |
"""

DEFTER_KUNYE = """Bu dosya `pct_v2/DESCRIPTION_EN.md` ve `DESCRIPTION_TR.md` içinde geçen **her sayının** ham
kaynak satırını tutar (künye biçimi `dosya:satır`). Her künye, sayının geçtiği ham satır
okunarak doğrulandı; doğrulama tekrar üretilebilir: ilgili satırı `sed -n 'Np' <dosya>`
komutuyla açın."""


def blok(path: Path, bas: str, bit: str, yeni: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.startswith(bas))
        j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith(bit))
    except StopIteration:
        raise SystemExit(f"{path.name}: {bas!r} / {bit!r} sınırları bulunamadı")
    path.write_text("\n".join(lines[:i] + yeni.rstrip("\n").splitlines() + [""] + lines[j:]) + "\n",
                    encoding="utf-8")
    return [f"{path.name}: satır {i+1}-{j} -> tablo"]


def main() -> int:
    out = blok(NOT, "- **Fiziksel biçim (Rule 11.2", "## 6.", BES)
    out += blok(NOT, "**Vekile bırakılanlar", "## 7.", VEKIL)

    text = DEFTER.read_text(encoding="utf-8")
    old = """Bu dosya `pct_v2/DESCRIPTION_EN.md` ve `DESCRIPTION_TR.md` içinde geçen **her sayının** ham kaynak
satırını tutar. Sayılar 2026-11-19 tarihli oturumda `python3` ile satır satır okunarak doğrulanmıştır
(kontrol komutu: `sed -n 'Np' <dosya>` / dosya satırına erişim). künye: `dosya:satır`."""
    if text.count(old) != 1:
        raise SystemExit(f"OLCUM künye: beklenen 1 eşleşme, bulunan {text.count(old)}")
    DEFTER.write_text(text.replace(old, DEFTER_KUNYE), encoding="utf-8")
    out.append("OLCUM_KAYITLARI.md: künye satırı künye-biçimi tanımıyla yeniden kuruldu")

    print("\n".join(out))

    t = NOT.read_text(encoding="utf-8")
    for gerekli in ("YAPILMADI", "manual inspection required", "request form",
                    "PATENT_ENABLE_ANTECEDENT_CHECK=1", "ST.26", "bağımsız doğrulayıcı",
                    "proxy", "Vekile bırakılanlar", "## 6.", "## 7."):
        assert gerekli in t, f"içerik kaybı: {gerekli!r}"
    d = DEFTER.read_text(encoding="utf-8")
    assert "sed -n" in d and "2026-11-19" not in d.splitlines()[3]
    print(f"\nSAGLAMA OK: not {len(t.splitlines())} satır · defter {len(d.splitlines())} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
