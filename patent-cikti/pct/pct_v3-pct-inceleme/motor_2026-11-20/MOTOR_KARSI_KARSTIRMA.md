# Motor karşılaştırma — pct_v3 kayıtları vs pct_v2 tabanı (2026-11-20)

Ham kayıtlar bu dizinde (`*_v3.json`); üretim bu oturumda `kayit_sunucusu.py` üzerinden, girdiler ham
bayt olarak `docs/patent/pct_v3/filing/`'den. Taban: `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/`.
v2 kayıtları `{status,result}` sarmalı, v3 kayıtları ham araç çıktısı — karşılaştırma `result`
normalize edilerek yapıldı §. Aşağıdaki tabloda `v2/v3` hücreleri izlek: `v2 <sayı> · v3 <sayı>`;
ham `issues` listesi uzunlukları §1 altındaki satırlarda tek tek atflıdır.

## 1 — Kanal bazında sayılar

| Kanal | Araç | v2 iz | v3 iz | Yorum |
|---|---|---|---|---|
| A US istem (varsayılan) | review_patent_claims | 0 | 0 | aynı |
| B US istem antecedent AÇIK | review_patent_claims (env) | 43 | 41 | 2 düzeltme birebir yansıdı (istem 7 ve 8, `docs/patent/pct_v3/DEGISIKLIK_KAYDI_ISTEM.md` §1) |
| C US yazılı-tanım | review_specification | 47 | 17 | `compliant=true`; spec_coverage pct §2 |
| D US biçimsel | check_formalities | 2 CRITICAL | 2 CRITICAL | aynı: SUMMARY + DETAILED DESCRIPTION — kullanıcı kararıyla AÇIK (PCT başlık pratiği) |
| E EPC istem EN | review_epo_claims | 1 | 1 | aynı: conciseness → hukuki kalem, `docs/patent/pct_v3/ALTERNATIF_ISTEM_KUMESI.md` |
| F EPC yeterlilik EN | review_epo_specification | 47 | 17 | `compliant=true` |
| H EPO biçimsel | check_epo_formalities | skip | skip | `claims_fees` ATLANDI (araç istem gövdesini görmüyor) — ücret ölçülmedi |
| K PCT biçimsel | check_pct_formalities | 1 CRITICAL | 1 CRITICAL | `No claims detected` — araç sınır §3.2, metinle giderilemez |
| L paket bütünlüği | check_package | 0 | 0 | aynı: 6 belge, paket skoru 100 |
| ETR EPC istem TR | review_epo_claims | 16 | 16 | aynı — İngilizce-anahtar artefakt §3.3 |
| FTR EPC yeterlilik TR | review_epo_specification | 0 | 0 | aynı |

Tek tek ham atıflar — v2 `issues` uzunluğu ile v3 karşılığı: A: `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/A_us_claims_en_default.json:9` ↔ `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/A_us_claims_en_default_v3.json:5`.
B: `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json:9` ↔ `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON_v3.json:9`.
C: `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/C_us_spec_support_en.json` ↔ `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/C_us_spec_support_en_v3.json:6`.
F: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/F_epo_spec_support_en_v3.json:7` (EPC eş-değeri).
E: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/E_epo_claims_en_v3.json:9`.
L: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/L_package_filing_v3.json:3` (toplam 0) ve
`patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/L_package_filing_v3.json:10` (files_scanned 6).
D: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/D_us_formal_abstract_filingcopy_v3.json:69` ve
`patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/D_us_formal_abstract_filingcopy_v3.json:79`. K: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/K_pct_formal_en_v3.json:97`.
ETR: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/ETR_epo_claims_tr_v3.json` (issues listesi §1 tablosu) ·
FTR: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/FTR_epo_spec_support_tr_v3.json` (boş liste §).

## 2 — C/F: 47 → 17 iz dökümü (element bazlı, `key=(claim, element)`) §

- **Aynı kalan 7 §:** istem 1 `point in a D-dimensional space` ve `tabular record`; istem 5
  `context length of the task input`; istem 10 `sentence cloud state`; istem 12 §
  `reader by gradient-based backpropagation`; istem 13 ve 15 `tabular record` — hepsi §
  `severity=IMPORTANT`, `confidence=MEDIUM`, tek-paragraf-atıflı (`patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/C_us_spec_support_en_v3.json:14` gibi).
- **Kaybolan 40 §:** v2'deki `task input`, `reader`, `point cloud`, `ordered sequence of`,
  `input order`, `context-encoded representations` gibi tekrarlı izler § — eklenen `[0071]–[0098]`
  bölümü bu unsurları paragraf düzeyinde destekleyince kayboldu §.
- **Yeni kelime-varlıkları 10 §:** `point cloud of`, `clustering operation`, `pair-geometry features`,
  `window-geometry encoder`, `interpretation engine`, `context encoder` — hepsi IMPORTANT/MEDIUM,
  tek-paragraf-atıf eşiğinde; destek-eksikliği değil **eşik-hassasiyeti** izidir (iddia: `compliant` doğru
  `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/C_us_spec_support_en_v3.json:2` ve kapsama yüzdesi 100 `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/C_us_spec_support_en_v3.json:12`).
- Kalan 17'nin tamamı **vekilin kalem işi** — otomatik metin değişikliği önerisi yapılmadı §.

## 3 — Değişmeyen bilinen sınırlar (metinle giderilmez, kayıt) §

1. D 2 CRITICAL `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/D_us_formal_abstract_filingcopy_v3.json:69` — ABD başlık pratiği; PCT hedefi nedeniyle kullanıcı kararıyla açık (`docs/patent/pct_v3/KURULUM_PLAN.md` §2).
2. K `No claims detected` § `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/K_pct_formal_en_v3.json:97` + H `claims_fees` atlandı `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/H_epo_formal_en_v3.json:14` — araç imzası istem gövdesi almıyor; ücret **ölçülmedi**.
3. § TR metinde İngilizce anahtar arayan artefakt: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/ETR_epo_claims_tr_v3.json:9` (toplam 16); hukuki hüküm değil.

## 4 — Ham çıktı adresleri §

- `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/B_kosum.log:2` (toplam 41 · kritik 0) ve exit 0 kaydı `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/B_kosum.log:1`.
- **Not § (2026-11-20 akşamı, kullanıcı kararı (a)):** yukarıdaki v2 taban ölçümleri (B 43 — `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json:9` · C 47 — `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/C_us_spec_support_en.json:7`) sabahki v2 metnine aittir; v2 akşam 84 paragraflık adayla değişti (`patent-cikti/pct/pct_v3-pct-inceleme/dogrulama_v2kabul_2026-11-20.txt:2`). Ham JSON'lara dokunulmadı; v3 sayıları kendi künyelerine bağlıdır ve değişmemiştir.
- v3 B sayaç bloğu: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON_v3.json:468-470` (claim_count 15 / independent_count 3 / dependent_count 12) — v2 ile aynı yapı `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json:490-492`.
- Sunucu günlüğü: `patent-cikti/pct/pct_v3-pct-inceleme/motor_2026-11-20/sunucu.log` (port 8767, girdi kökü `docs/patent/pct_v3/filing`).
