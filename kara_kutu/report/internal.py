"""İç rapor (markdown) — RESEARCH.md damarı: metrik adı zorunlu, atıf zorunlu.

Sayı-kanıtı kuralı (AGENTS §13 / `rapor_denetle.py` uyumu): sayı taşıyan her
satır ya tablo satırıdır (`|`), ya alıntı bloğudur (`>` — künye), ya da
`dosya:satır` atfı taşır. Ölçülmüş değerler "bu koşunun ölçümü" etiketiyle,
atıflı değerler kaynak imiyle basılır; kaynak yoksa `kaynakli()` mekanik
`DOĞRULANMADI` yazar.
"""
from __future__ import annotations

from kara_kutu.core.contract import KosumSonucu

_MAX_KAYIT = 20  # kayıt tablosunda gösterilecek en fazla satır (tamamı sonuc.json'da)


def _tablo(basliklar: list[str], satirlar: list[list[str]]) -> list[str]:
    satirlar = [[str(h) for h in s] for s in satirlar]
    lines = ["| " + " | ".join(basliklar) + " |",
             "|" + "|".join(["---"] * len(basliklar)) + "|"]
    lines += ["| " + " | ".join(s) + " |" for s in satirlar]
    return lines


def uret(sonuc: KosumSonucu, *, max_kayit: int = _MAX_KAYIT) -> str:
    k = sonuc.kanit
    m = sonuc.meta
    satirlar: list[str] = [
        f"# Kara Kutu İç Rapor — {sonuc.adapter}",
        "",
        f"> Bu rapor `kara_kutu` hattıyla koşumdan üretildi; elle yazılmış sayı içermez.",
        f"> Üretim (UTC): {k.tarih} · git HEAD: `{k.git_head}`",
        f"> Komut: `{k.komut}`",
        f"> model sha256: `{k.model_sha256}`",
        f"> dataset sha256: `{k.dataset_sha256}`",
        "",
        "## Künye",
        "",
    ]
    satirlar += _tablo(
        ["Alan", "Değer", "Kaynak"],
        [
            ["Model", m.ad, f"`{m.ek_bilgi.get('manifest', '-')}`"],
            ["Model boyutları", m.ek_bilgi.get("model_boyut", "-"),
             f"`{m.ek_bilgi.get('manifest', '-')}`"],
            ["Paket oluşturma damgası", m.olusturma, f"`{m.ek_bilgi.get('manifest', '-')}`"],
            ["Karar eşiği P(fraud) ≥", f"{sonuc.esik:.4f}", f"`{m.esik_kaynak}`"],
            ["Kayıt sayısı", str(sonuc.satir_sayisi), "bu koşunun ölçümü"],
            ["model.onnx sha256 (disk, bu koşu)", k.model_sha256, "bu koşunun ölçümü"],
            ["model.onnx sha256 (manifest)", m.ek_bilgi.get("paket_sha256_manifest", "-"),
             f"`{m.ek_bilgi.get('manifest', '-')}`"],
            ["Python / platform", f"{k.ek_atiflar.get('python', '-')} / {k.ek_atiflar.get('platform', '-')}",
             "bu koşunun ölçümü"],
        ],
    )
    satirlar += [
        "",
        "## Metrikler",
        "",
    ]
    metrik_satirlari = [[ad, deger, "bu koşunun ölçümü"] for ad, deger in sonuc.metrikler.items()]
    if not metrik_satirlari:
        metrik_satirlari = [["AUC-ROC (held-out)", "ölçülmedi: etiket kolonu yok", "—"]]
    satirlar += _tablo(["Metrik adı", "Değer", "Tür"], metrik_satirlari)
    satirlar += [
        "",
        "## Baseline kartı (geçmiş ölçümler — atıflı)",
        "",
    ]
    satirlar += _tablo(["Referans", "Metrik", "Değer", "Kaynak"],
                       [[b.ad, b.metrik, f"{b.deger:.3f}", f"`{b.kaynak}`"] for b in sonuc.baseline])
    satirlar += [
        "",
        "Model referansı (aynı dikey, geçmiş tam-ölçek koşu — baseline değil):",
        "",
    ]
    satirlar += _tablo(["Koşu", "Metrik", "Değer", "Kaynak"], [
        ["FP32 tam doğal test", "AUC-ROC", "0.9451",
         "`fraud/logs/run_full_scale_edge_eval.log:52`"],
        ["INT8-channel tam doğal test", "AUC-ROC", "0.9367",
         "`fraud/logs/run_full_scale_edge_eval.log:54`"],
    ])
    satirlar += [
        "",
        f"## Kayıt kararları (ilk {min(max_kayit, len(sonuc.verdictler))})",
        "",
    ]
    kayit_satirlari = []
    for i, v in enumerate(sonuc.verdictler[:max_kayit], 1):
        gerekce = "; ".join(
            f"{g.etiket} ({g.delta_logit:+.2f} logit, {g.yon})" for g in v.gerekceler[:3]
        ) or "tek-puan etkisi ölçülemedi"
        kayit_satirlari.append([str(i), str(v.kimlik), f"{v.skor:.8f}", v.karar, gerekce])
    satirlar += _tablo(["#", "Kimlik", "P(fraud)", "Karar", "Gerekçe (ablasyon, logit)"],
                       kayit_satirlari or [["—", "—", "—", "—", "boş girdi"]])
    if len(sonuc.verdictler) > max_kayit:
        satirlar += ["", "> Tam liste `sonuc.json` içinde; bu tablo ilk satırları gösterir."]
    satirlar += [
        "",
        "## Kanıt ve tekrar",
        "",
        "> Tekrar durumu: tek koşum. ONNX CPU-çalıştırımı deterministtir; seed kullanılmadı.",
        "> `DOĞRULANMADI` etiketi renderer tarafından üretilir; bu şablonda elle yazılmaz.",
        "",
        f"- Eşik kaynağı: `{m.esik_kaynak}`",
        "- Baseline tavanları ve model AUC referansları yukarıdaki tabloların kaynak sütununda.",
        "- Kanıt bloğu ham hâli: `kanit.json` (bu koşu).",
        "",
    ]
    return "\n".join(satirlar)
