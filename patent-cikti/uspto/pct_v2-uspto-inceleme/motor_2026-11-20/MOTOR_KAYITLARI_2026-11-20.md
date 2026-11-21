# MOTOR_KAYITLARI_2026-11-20 — koşum künyesi (her sayı bir ham kayıt dosyasına bağlı)

- **Tarih:** 2026-11-20 · **Motor:** `patent-creator` MCP sunucusu (Claude-Patent-Creator, yerel)
- **Sunucu tanımı:** `~/.config/opencode/opencode.jsonc` → `patent-creator` =
  `/Users/serkankose/patent-creator-venv/bin/python …/site-packages/mcp_server/server.py` (`local`, `enabled: true`)
- **Denetlenen metin:** PCT v2 **gönderim nüshaları** (`docs/patent/pct_v2/filing/`) — kaynak dosyalar değiştirilmedi.
- **Girdi aktarımı:** metin elle yeniden yazılmadı. Bu dizindeki kayıt sunucusu betiği
  (`kayit_sunucusu.py`) yalnız yerel adreste dinledi — adres ve port bir alt satırda. Betik
  gönderim nüshalarını HAM BAYT olarak sundu; Code Mode `fetch` ile okundu ve araç argümanına bayt-bayt
  verildi. Adres: `127.0.0.1`, port `8731`.
  Araç yanıtları yine POST ile bu dizine ham JSON olarak yazıldı (`/save?name=…`).
- **Başlık argümanı:** gönderim istem dosyasındaki `Title of the invention:` satırından türetildi
  ("Dual-use point-cloud representation with embedded explainability", 64 karakter).
- **`drawings_present=true` gerekçesi:** `docs/patent/pct_v2/filing/SEKILLER_FILING-COPY.pdf` (8 şekil) +
  `docs/patent/pct_v2/fig/` SVG'leri. Araç bunu **beyan** sayar: "drawings_present is a declaration, not
  inspection of drawing files" (`D_us_formal_abstract_filingcopy.json` `verdict_scope`).

## Girdi künyesi (ölçüm: `shasum -a 256` + `wc -c`)

| Girdi | Bayt | sha256 (ilk 16) |
|---|---|---|
| `filing/CLAIMS_FILING-COPY_EN.txt` | 10.988 | `38ce1f92180f8507` |
| `filing/DESCRIPTION_FILING-COPY_EN.txt` | 36.677 | `a228276a14fc6028` |
| `filing/ABSTRACT_FILING-COPY_EN.txt` | 864 | `4157b9f924619f9e` |
| `filing/ISTEMLER_FILING-COPY_TR.txt` | 11.715 | `50160803c4f77624` |
| `filing/TARIFNAME_FILING-COPY_TR.txt` | 37.279 | `fff62fd35037fa5f` |
| `filing/OZET_FILING-COPY_TR.txt` | 828 | `1ce74faf76668ad5` |

## Koşan araçlar ve ham kayıt dosyaları

| # | Araç (MCP) | Girdi | Ham kayıt |
|---|---|---|---|
| A | `review_patent_claims` | EN istemler | `A_us_claims_en_default.json` |
| B | `ClaimsAnalyzer().analyze_claims` — opt-in `antecedent_basis` AÇIK | EN istemler | `B_us_claims_en_antecedentON.json` |
| C | `review_specification` | EN istemler + EN tarifname | `C_us_spec_support_en.json` |
| D | `check_formalities` | başlık + EN özet + EN tarifname + `drawings_present=true` | `D_us_formal_abstract_filingcopy.json` |
| E | `review_epo_claims` | EN istemler | `E_epo_claims_en.json` |
| F | `review_epo_specification` | EN istemler + EN tarifname | `F_epo_spec_support_en.json` |
| H | `check_epo_formalities` | başlık + EN özet + EN tarifname + şekil beyanı | `H_epo_formal_en.json` |
| K | `check_pct_formalities` | başlık + EN özet + EN tarifname + şekil beyanı | `K_pct_formal_en.json` |
| L | `check_package` | `docs/patent/pct_v2/filing/` dizini | `L_package_filing.json` |
| E-TR | `review_epo_claims` | TR istem nüshası | `ETR_epo_claims_tr.json` |
| F-TR | `review_epo_specification` | TR istem + TR tarifname | `FTR_epo_spec_support_tr.json` |

**B satırının doğası (gizlenmez):** MCP aracının bayrağı yok — `review_patent_claims` `antecedent_basis`
kontrolünü varsayılan KAPALI koşar ve `checks_skipped` içinde `enable_with: PATENT_ENABLE_ANTECEDENT_CHECK=1`
diyor (`A_us_claims_en_default.json`). Ortam değişkeni MCP sunucusu sürecine yüklenemediği için aynı sınıf,
aynı venv python'i ile ve girdi diskten HAM okunarak koşuldu (`B_antecedent_kos.py`; MCP sarmalayıcısının
çağrısıyla aynı sınıf/metot: `mcp_server/tools/analyzer_tools.py:81-83`). Bu satır MCP aracı değil,
**motor sınıfının opt-in varyantı**dır.

## Yeniden üretim

```
python3 kayit_sunucusu.py <port>                   # girdi sunucusu (bu dizin; bu koşumda yerel port kullanıldı)
PATENT_ENABLE_ANTECEDENT_CHECK=1 \
  /Users/serkankose/patent-creator-venv/bin/python B_antecedent_kos.py
/Users/serkankose/patent-creator-venv/bin/python uretim_metrik.py   # METRIKLER tablosu
```

`uretim_metrik.py` yeniden koşulduğunda `METRIKLER_2026-11-20.md` bayt-bayt aynı üretildi
(`diff` boş) — tablodaki her sayı ham kayda bağlıdır.

`rapor_doldur.py` raporu `@@İŞARET@@` yerlerine ham kayıt metinlerini koyarak doldurur; işaretler
tüketildiği için ikinci koşum kendi `assert`'uyla durur (dosyayı bozmaz). İstisna: §3.4'teki
"araç içi tutarsızlık" notu elle yazıldı ve dayanağı `H_epo_formal_en.json` içinde yan yana duran iki
alanın kendisidir (`summary` → "Found 1 WARNING…" · `issues[0].severity` → `IMPORTANT`).

## Gözlem komutları (bu turda koşulanlar, ham çıktılarıyla)

| Komut | Ham çıktı |
|---|---|
| `grep -n karakterize docs/patent/pct_v2/filing/ISTEMLER_FILING-COPY_TR.txt` | üç satır: `5`, `29`, `33` (bağımsız istem 1, 13, 15 satırları) |
| `grep -i -w claim docs/patent/pct_v2/filing/ISTEMLER_FILING-COPY_TR.txt` | eşleşme yok; `grep -c` sayaç olarak `0` döndü ve çıkış kodu 1 (eşleşme yok) |
| `sed -n '21p;76p' H_epo_formal_en.json` | `21` → `summary` "… Found 1 WARNING EPO formality issue(s)"; `76` → `"severity": "IMPORTANT"` |
| `sed -n '167p;210,213p' …/mcp_server/epo_claims_analyzer.py` | bağımlılık regex'i ve iki-parça kalıpları (rapor §3.5'te alıntılandı) |
| `sed -n '352,356p' …/mcp_server/tools/epo_analyzer_tools.py` | `check_pct_formalities` imzası: `abstract`, `title`, `specification`, `drawings_present` |
| `shasum -a 256` + `wc -c` (6 gönderim nüshası) | yukarıdaki "Girdi künyesi" tablosu |

## Kanalların durumu (araç `check_*_status` çıktıları — ham kayıtlar bu dizinde)

| Ham kayıt | Araç | Alan | Anlam |
|---|---|---|---|
| `Z_durum_google_patents.json` | `check_google_patents_status` | `ready: false` | SerpApi anahtarı yok |
| `Z_durum_bigquery.json` | `check_bigquery_status` | `available: false` | Google Cloud proje kimliği yok |
| `Z_durum_epo_api.json` | `check_epo_api_status` | `ready: false` · `credentials_configured: false` | EPO OPS kimlik bilgisi yok |
| `Z_durum_diagram.json` | `check_diagram_tools_status` | `ready: true` · Graphviz 16.1.0 | şekil araçları çalışıyor |

Bu nedenle bu raporda **yeni bir ön-sanat/patent veritabanı taraması yapılmadı**; tarama bir önceki turda
tek kanalla (Google Patents vekil proxy) koştu ve kaydına `docs/patent/pct_v2/ON_SANAT.md` §5'te duruyor.
Bu rapordaki tüm bulgular **başvuru metninin otomatik denetimleridir**; ön-sanat/yenilik iddiası içermez.

## Makine kapıları (ham çıktı `KAPI_OUT_2026-11-20.txt`, üretici `uretim_kapi.sh`)

- Rapor sayı denetçisi bu üç dosyada **kaynakta yok 0 · ölü adres 0** verdi; kabul ölçütü budur.
  Kalan "kaynaksız" satırlar, aşağıdaki "sayıların bağlama biçimi" açıklamasındaki JSON-alanı bağlamalı
  satırlardır. Güncel sayaçlar `KAPI_OUT_2026-11-20.txt` içindeki ham çıktıdadır (bu dosya sayıyı
  kopyalamıyor — metin bayatlamasın diye sayaç yazılmadı).
  Çıkış kodu `2` = yalnız uyarı.
- Küresel `denetim_kos.py hepsi` koşumunun ham ÖZET satırı ve kayıt yolu bir alt satırda.
  Sonuç: bulgu 12 · uyarı 7 · bilgi 29 · çıkış kodu 1. Kayıtta bu depoya ait BULGU satırı **yok**
  (tamamı başka depolardan). Bu koşumun dosyaları `patent-cikti` altında ve gitignore'da olduğu için
  küresel taramanın rapor sınıfına girmiyor; bu yüzden denetçi doğrudan dosya yollarıyla ayrıca koşuldu.



## Sayıların bağlama biçimi (gizlenmez)

- Sayılar bu raporda `dosya:satır` değil, **kayıt dosyası + JSON alanı** biçimiyle bağlıdır
  (ör. "`C` kaydı `written_description_issues`"). Sebep: sayıların kaynağı satır değil JSON alanıdır.
  Alan alan künye aynı dizindeki `METRIKLER_2026-11-20.md` tablosundadır.
- Makine kapısı (`python3 ~/.config/opencode/denetim/rapor_denetle.py <bu rapor>`) bu biçimi
  `dosya:satır` grameriyle okur; bu yüzden iz satırları "kaynaksız" sayılıyor. "kaynaksız" sayılan
  satırların tamamı ya ham araç özetinin alıntılandığı satırlar ya da `C`/`B` kayıtlarından sırayla
  kopyalanan bulgu izleri — hepsi bölüm başlığında adı verilen kayıt dosyasından
  `rapor_doldur.py` ile elle yazılmadan doldurulmuştur. Güncel sayaçlar `KAPI_OUT_2026-11-20.txt`'de.
- Bağımsız doğrulayıcı turu ( `reviewer` alt-ajanı) A–J maddelerinde **10 ✓ / 0 ✗ / 1 KANITSIZ** dedi;
  tek kanıt eksiği `check_google_patents_status → ready: false` cümlesinin ham kaydının dizinde
  bulunmamasıydı. Bu kayıt sonradan `Z_durum_google_patents.json` adıyla diske alındı (üstteki tablo).
