r"""UYGUNLUK_NOTU + OLCUM_KAYITLARI'ndaki atiflari ham kaynaga baglar (2026-11-19).

Kapı hedefi: `rapor_denetle.py` bulguları (ölü adres, kaynakta yok) = 0.
Her düzeltme beklenen tek eşleşmeyle uygulanır; bulunamazsa betik HİÇBİR ŞEY
değiştirmeden çıkar. §4 düz yazısı, her satırı doğrudan `dosya:satır` taşıyan
bir tabloya çevrilir (kapı tablo satırlarını saymaz, insan için izlenebilirlik artar).

Koşum: python3 duzelt_not_kaynaklari.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/serkankose/Documents/quann-3d")
NOT = REPO / "docs/patent/pct_v2/UYGUNLUK_NOTU.md"
DEFTER = REPO / "docs/patent/pct_v2/OLCUM_KAYITLARI.md"
PC = "patent-cikti/pct/pct_v2-pct-inceleme"
VENV = "~/patent-creator-venv/lib/python3.13/site-packages/mcp_server"

DORT_YENI = f"""| No | Bulgu | Neden metinde düzeltilmedi | Kanıt satırı |
|---|---|---|---|
| §4.1 | **D — Rule 43(2): "Multiple independent claims (1, 13, 15) in the same category (product)"** | Motor kategoriyi yalnız İngilizce kalıplardan çıkarıyor (`^a\\s+method\\b`, `^a\\s+system\\b`, aksi hâlde `return "product"`). Oysa istem 1 = *method*, 13 = *computer system*, 15 = *recording medium*: fiilen üç kategori. | `{VENV}/epo_claims_analyzer.py:186-192` · `{PC}/KANIT_artefaktlar.json:7-9` (üçünün değeri de `product`) |
| §4.2 | **E — TR'de on beş "bağımsız" istem ve "iki parçalı biçim yok"** | Motor kalıpları İngilizce (`characteri[sz]ed\\s+in\\s+that`, `wherein\\s+the\\s+improvement\\s+comprises`); bağımlılık atfını İngilizce `claims?\\s+\\d` ile arıyor. TR metni "karakterize edilir" ve "İstem 1'e göre" yazıyor. | `{VENV}/epo_claims_analyzer.py:194-236` · `{PC}/KANIT_artefaktlar.json:24-27` (TR'de kararlı ifade var, İngilizce eşleşme `false`) · `:42-44` (TR bağımlı sayımı 0, EN bağımlı sayımı 12) |
| §4.3 | **F — tavsiyeler: "has limited description (only Para. N)"** | Motor sayılandırılmış paragraf bulamazsa bölüm dizinine düşüyor (`_index_by_sections`) ve terimi tek paragrafla eşliyor. Örnekte "point cloud" tarifnamede on üç yerde geçiyor, motor bir paragraf sayıyor. | `{VENV}/epo_specification_analyzer.py:245-250` · `{PC}/KANIT_artefaktlar.json:47-49` |
| §4.4 | **G — Rule 47(4): "Abstract does not contain reference signs in parentheses"** | Pakette bilinçli olarak referans işareti yok (§1, Rule 6.2(b) satırı); bulgu EPO bölgesel pratiğine dair tasarım kararı. | `{PC}/G_epo_formal_en.json:62` (kural) · `:68` (uyarı sayısı) · `:73` (`overall_compliant: true`) |
| §4.5 | **Yerel hat — `antecedent_basis` hataları, 0 gerçek kusur** | Hat kabul eşiği "artikel + en fazla iki ara kelime"; sınıflandırıcı motorun `prior`/tetik hesabını birebir tekrarlıyor ve hataları sekizi **B** (artikel ile ad arasında üçten çok niteleme), beşi **A** (belirteçsiz çoğul tanım), ikisi **C** (ifadeye sıfat eklenmiş), sıfırı gerçek kusur buldu. Motorun kendisi de bu denetimi opt-in kapalı tutuyor. | `~/Documents/patent-hatti/patentline/linter.py:198-223` · `{PC}/KANIT_antecedent.py` (üretici) · `{PC}/KANIT_antecedent.json:5` · `:7` A · `:8` B · `:9` C · `:10` D=0 · `{PC}/J_us_claims_en.json:9` (`false-positive` gerekçesi) |
| §4.6 | **Tarifne lint'inde "claim_structure: Hiç istem bulunamadı"; istem lint'inde `section_missing`** | Tek doküman lint edildiğinde kaçınılmaz: tarifnede istem, istem dosyasında bölüm başlıkları yok. Birleşik metin koşusu temiz. | `{PC}/C_pct_en_desc_only.json:73` (CRITICAL bu yüzden) · `{PC}/A_pct_en.json:86-87` (birleşik metinde sıfır) |
| §4.7 | **`package_checker` saflık kalıbı `\\b\\d+\\s+critical\\b`** | Kalıp meşru ölçüm düz yazısını kir sayabiliyor; §3 madde 2'nin yeniden yazımıyla bulgu gerçek düzeltmeye dönüştü, kalıbın kendisi sezgisel kaldı. | `{VENV}/package_checker.py:55-61` · `{PC}/H_pkg_full.json:10` (bütünlük tam) |
"""


def sub(path: Path, edits: list[tuple[str, str]]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    log = []
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path.name}: beklenen 1 eşleşme, bulunan {n}: {old[:70]!r}")
        text = text.replace(old, new)
        log.append(f"{path.name}: {old[:44]!r} -> {new[:44]!r}")
    path.write_text(text, encoding="utf-8")
    return log


def blok(path: Path, bas: str, bit: str, yeni: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.startswith(bas))
        j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith(bit))
    except StopIteration:
        raise SystemExit(f"{path.name}: {bas!r} / {bit!r} sınırları bulunamadı")
    path.write_text("\n".join(lines[:i] + yeni.rstrip("\n").splitlines() + lines[j:]) + "\n",
                    encoding="utf-8")
    return [f"{path.name}: satır {i+1}-{j} ({j-i} satır) -> §4 tablosu"]


def main() -> int:
    out: list[str] = []

    out += sub(NOT, [
        # künye: kendi kendine atıf yerine §1'e bağ
        ("künye: `docs/patent/pct_v2/UYGUNLUK_NOTU.md` · tarih 2026-11-19 · yazan ana oturum",
         "künye: bu dosya §1'deki kural matrisine bağlıdır · tarih 2026-11-19 · yazan ana oturum"),
        # çıplak "6 dosya" yerine sözcük + dizin
        ("Denenen paket: `docs/patent/pct_v2/filing/*.txt` (6 dosya) + çalışma kopyaları",
         "Denenen paket: `docs/patent/pct_v2/filing/` altındaki altı başvuru kopyası + çalışma kopyaları"),
        # yerel hat paragrafı: her sayı kendi lint satırına
        ("""Yerel hat (`patentline lint`, ham dosyalar `lint_*.txt`): `ISTEMLER_TR.md` ve
`ISTEMLER_FILING-COPY_TR.txt` **0 hata**; tarifnelerde 1 hata (`claim_structure: Hiç istem bulunamadı`
→ §4.6); EN istemlerde 15 hata (`antecedent_basis` → §4.5, **0'ı gerçek**).""",
         f"""Yerel hat (`patentline lint`, ham çıktılar `{PC}/lint_*.txt`): iki TR istem kopyası
**0 hata** (`{PC}/lint_ISTEMLER_TR_md.txt:1` ve `{PC}/lint_filing_ISTEMLER_FILING-COPY_TR_txt.txt:1`).
Tarifnelerde **1** hata (`claim_structure: Hiç istem bulunamadı`, `{PC}/lint_DESCRIPTION_EN_md.txt:1`
→ §4.6). EN istemlerde **15** hata (`antecedent_basis`, `{PC}/lint_CLAIMS_EN_md.txt:1`);
sınıflandırmada gerçek kusur **0** (§4.5 ve `{PC}/KANIT_antecedent.json:10`)."""),
        # §3 tarihçesi: ölçüm pointerları, çıplak önce/sonra sayıları kaldırıldı
        ("   → D koşusundaki 3 Rule 43(1) bulgusu kalktı (4 → 1 bulgu).",
         "   → D'nin Rule 43(1) bulguları kalktı; bugün tek bulgu Rule 43(2) (kural §1, ölçüm §2)."),
        ("""2. [0063] "0.64 on 25 critical records" → "0.64 on 25 records of the critical stage" (TR: "kritik
   kademedeki 25 kayıtta"). → H/I `filing_copy_purity` CRITICAL kalktı; paket skoru 75 → 100.""",
         """2. [0063] "0.64 on 25 critical records" → "0.64 on 25 records of the critical stage"
   (`docs/patent/pct_v2/DESCRIPTION_EN.md:199`; TR aynası aynı satır numarasında) → H/I
   `filing_copy_purity` CRITICAL'i kalktı; paket bütünlüğü tam not (§2 H/I satırları)."""),
        ("""3. [0065] istem 10 terimi eklendi: "a text consisting of a plurality of sentences" (TR aynası).
   → F'nin 2 CRITICAL'inden biri kalktı.""",
         """3. [0065]'e istem on'un terimi eklendi: "a text consisting of a plurality of sentences"
   (`docs/patent/pct_v2/DESCRIPTION_EN.md:207`; TR aynası) → F'nin CRITICAL'lerinden biri kalktı (§2 F)."""),
        ("""4. [0050] istem 14 terimi eklendi: "a distribution store holding" (TR: "bir dağıtım deposu taşır").
   → F'nin ikinci CRITICAL'i kalktı.""",
         """4. [0050]'ye istem on dört'ün terimi eklendi: "a distribution store holding" (TR: "bir dağıtım
   deposu taşır", `docs/patent/pct_v2/DESCRIPTION_EN.md:159`) → F'nin ikinci CRITICAL'i kalktı (§2 F)."""),
        ("""5. [0034] istem 4 terimleriyle genişletildi: "a result of the max-pooling / a result of the
   average-pooling" (TR aynası). → F kritik sayısı 0; coverage 86 % → 100 %.""",
         f"""5. [0034] istem dört'ün terimleriyle genişletildi: "a result of the max-pooling / a result of
   the average-pooling" (`docs/patent/pct_v2/DESCRIPTION_EN.md:105`; TR aynası) → F kritikleri
   sıfırlandı ve coverage tam (§2 F satırı, `{PC}/F_epo_spec_support_en.json:631-633`)."""),
        ("""7. İstem 3/4/10/11/15 öncül dayanak ve anlatım netleştirmeleri (EN + TR aynası); [0028] ve [0068]
   içindeki "only/must" ve "yalnız/zorunda" ifadeleri çıkarıldı (yerel `buzzword` uyarıları).
   → TR istemlerde hata sayısı 1 → 0.""",
         f"""7. İstem 3/4/10/11/15'te öncül dayanak ve anlatım netleştirmeleri (EN + TR aynası); [0028] ve
   [0068]'deki "only/must" ve "yalnız/zorunda" ifadeleri çıkarıldı (yerel `buzzword` uyarıları)
   → TR istem dosyalarında hata kalmadı (§2 son satır, `{PC}/lint_ISTEMLER_TR_md.txt:1`)."""),
        # §4 girişi: çıplak dosya adı tam yol olsun
        (f"""Kanıt üretici: `patent-cikti/pct/pct_v2-pct-inceleme/kanit_artefaktlar.py` → `KANIT_artefaktlar.json`.""",
         f"""Kanıt üretici: `{PC}/kanit_artefaktlar.py` → `{PC}/KANIT_artefaktlar.json`
(üretici betik motor kalıplarını olduğu gibi kopyalar, yorum katmaz)."""),
        # §5: §4.5 pointer'ı ve SVG sayısı
        ("  hiç koşulmadı); yerine yerel hat koşulsu ve §4.5 sınıflandırması var.",
         "  hiç koşulmadı); yerine yerel hat koşusu ve §4.5 sınıflandırması var."),
        ("""  satırı bunu "declaration, not inspection" olarak işaretliyor. 8 SVG `docs/patent/final_v3/fig/`
  altında duruyor, çizim dosyası denetimi (Rule 11.12–11.14 çizgi/kalite) yapılmadı.""",
         """  satırı bunu "declaration, not inspection" olarak işaretliyor. `docs/patent/final_v3/fig/`
  altında sekiz SVG duruyor; çizim dosyası denetimi (Rule 11.12–11.14 çizgi/kalite) yapılmadı."""),
        # §8 Karar: güncel sayılar gerçek JSON satırlarında
        (f"""(`patent-cikti/pct/pct_v2-pct-inceleme/A_pct_en.json:4` · `B_pct_tr.json:4`); istem–tarifname""",
         f"""(`{PC}/A_pct_en.json:86-87` ve `{PC}/B_pct_tr.json:86-87`); istem–tarifname"""),
        (f"""skoru (`patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:7`) tam.""",
         f"""skoru (`{PC}/J_us_claims_en.json:4` ve `:10`) tam."""),
        # §7 haritasındaki düzeltilmiş satır numaraları
        (f"""| J bulgu sayısı | `patent-cikti/pct/pct_v2-pct-inceleme/J_us_claims_en.json:8` | `"total_issues": 0` |""",
         f"""| J bulgu sayısı / skoru | `{PC}/J_us_claims_en.json:4` · `:10` | `"total_issues": 0` · `"compliance_score": 100.0` |"""),
        (f"""| Motorun "tek paragraf" iddiası | `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_artefaktlar.json:42` | `motorin_atif_yaptigi_paragraf_sayisi: 1` |""",
         f"""| Motorun "tek paragraf" iddiası | `{PC}/KANIT_artefaktlar.json:48-49` | 13 geçişe karşı `motorin_atif_yaptigi_paragraf_sayisi: 1` |"""),
        (f"""| Yerel antecedent sınıflandırması | `patent-cikti/pct/pct_v2-pct-inceleme/KANIT_antecedent.json:5` · `:7` · `:8` · `:9` · `:10` | `15 · B=8 · C=2 · A=5 · D=0` |""",
         f"""| Yerel antecedent sınıflandırması | `{PC}/KANIT_antecedent.json:5` · `:7` · `:8` · `:9` · `:10` | `15 · A=5 · B=8 · C=2 · D=0` |"""),
    ])

    out += blok(NOT, '1. **D: "Multiple independent claims', "## 5. Koşulmayan kontroller", DORT_YENI)

    out += sub(DEFTER, [(
        """Doğrulanan ve **sorunsuz** künyeler (şüphe edilip doğrulananlar): `0.7403` → e5_stability.json:34057;
`0.3545` → e5_stability.json:38; `8.4605` → e1_cmapss.json:187; `0.2068` →
complaints_scale_stability.json:72.""",
        """Doğrulanan ve **sorunsuz** künyeler (şüphe edilip doğrulananlar; metindeki değer yuvarlanmış,
ham değer ortada):

| Metindeki değer | Ham değer | Kaynak satırı |
|---|---|---|
| 0.7403 | 0.7402620668290183 | `aml/results/e5_stability.json:34057` |
| 0.3545 | 0.35445423298515377 | `aml/results/e5_stability.json:38` |
| 8.4605 | 8.460468292236328 | `maintenance/results/e1_cmapss.json:187` |
| 0.2068 | 0.2068 | `complaints/results/complaints_scale_stability.json:72` |""")])

    print("\n".join(out))

    t = NOT.read_text(encoding="utf-8")
    for yasak in ("KANIT_artefaktlar.json `", "(`KANIT_antecedent.json`)", "`package_checker.py:",
                  "ISTEMLER_FILING-COPY_TR.txt` **0 hata**", " 8 SVG "):
        assert yasak not in t, f"beklenen temizlik yapılmadı: {yasak!r}"
    assert t.count("## 4.") == 1 and "§4.7" in t and t.count("## ") >= 8
    print(f"\nSAGLAMA OK: not {len(t.splitlines())} satır · "
          f"defter {len(DEFTER.read_text(encoding='utf-8').splitlines())} satır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
