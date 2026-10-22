# VEKİLE GÖNDERİM NOTU — İki Başvuru Paketi

**Konu:** Düşük-Boyutlu Nokta-Bulutu Geometrisi temelli icatlar — (1) Türkiye ulusal patent + (2) PCT uluslararası başvuru
**Hazırlayan:** [İcatçı]  **Tarih:** 22 Ekim 2026
**Gizlilik:** Bu not ve ekleri TÜRKPATENT/PCT başvuru öncesi hazırlanmıştır; hiçbir kamuya açıklama yapılmamıştır.

---

## 1. Ne gönderiyoruz (2 dosya + çizimler)

### Dosya A — Türkiye ULUSAL patent (dil)
- **Teknik metin:** `docs/patent/TURKPATENT_basvuru_v2.3.md` (filing-ready: Başvuru Formu + İçindekiler + Açıklama + 19 İddia + Özet + Çizimler + Atıf Listesi + Vekile Not)
- **Kapsam:** Dil modelleme (bir-sonraki-token + agglütinatif morfoloji)
- **Başlık:** Düşük Boyutlu Nokta Bulutu Geometrisi Tabanlı Dil Modelleme Sistemi, Yöntemi ve Bilgisayar Programı
- **İddia:** 3 bağımsız (Sistem, Yöntem, Bilgisayar Programı) + 16 bağımlı = 19
- **Çizimler:** 8 SVG → `docs/figures/sekil_1..8_*.svg` (siyah-beyaz, referanslı, ayrı sayfa olarak PDF/TIFF render edilecek)
- **Öncelik:** T1 — bu ulusal başvurunun kendi tarihi

### Dosya B — PCT ULUSLARARASI başvuru (evrensel çekirdek + 4 dikey)
- **TR metin:** `docs/patent/geometry_dual_use_core_v1.md` (PCT/RO/101 + Açıklama + 13 İddia + Özet + Çizimler + Atıf + Vekil Değerlendirmesi)
- **EN metin:** `docs/patent/geometry_dual_use_core_v1_EN.md` (PCT uluslararası yayım için)
- **Kapsam:** Çift-geometri nokta-bulutu ile birleşik hesaplama + native açıklanabilirlik; 4 dikey embodiment (dil, fraud, AML, bakım)
- **Başlık:** Çift-Geometri Nokta-Bulutu Temsili ile Birleşik Hesaplama ve Native Açıklanabilirlik Yöneten Evrensel Sistem, Yöntemi ve Bilgisayar Programı
- **İddia:** 3 bağımsız + 10 bağımlı (6 mekanizma + 4 dikey) = 13
- **Çizimler:** Şekil 1-3 yeni SVG → `docs/figures/core_sekil_1..3_*.svg`; Şekil 4-7 dikey artifact figürlerinden (bkz. dosya içindeki tablo)
- **Öncelik:** T2 — bağımsız (bkz. §2)

---

## 2. Strateji — iki dosyanın ilişkisi (VEKİL KARARI)

| | Dosya A (TR ulusal) | Dosya B (PCT) |
|---|---|---|
| **Kapsam** | yalnızca **dil** | **evrensel** + dil/fraud/AML/bakım |
| **Öncelik** | T1 (kendi tarihi) | **T2 (kendi tarihi)** |
| **A'dan öncelik?** | — | **ALAMAZ** (evrensel kapsam A'da açıklanmadı) |

**Neden öncelik alamaz:** Öncelik, önceki başvurunun sonraki dosyanın kapsamını *açıklaması*nı gerektirir. A yalnızca dili açıklar; B'nin evrensel çekirdeği + fraud/AML/bakım A'da yoktur → B, A'dan öncelik **talep edemez**. İkisi **bağımsız** icat/kapsam olarak değerlendirilmelidir.

**Önerilen sıralama:**
1. **Önce Dosya A (TR ulusal, dil)** → düşük risk (tek dikey, teklik sorunu yok), erken öncelik T1, Türkiye'de koruma.
2. **Sonra Dosya B (PCT, evrensel+4)** → kendi önceliği T2; uluslararası kapsam (PCT'nin tümü veya hedef devletler vekil tercihine göre).

**Teklik (unity) notu (Dosya B):** 4 dikeyin tek başvuruda sunumu "çok icat" riski taşır. Savunma: tek icat = İddia 1'deki **dual-use geometri mekanizması** (aynı bulut hem skor hem native açıklama kaynağı); dikeyler bağımlı iddiadır. İncelemeci itiraz ederse dikeyleri **divisional**'a ayırma seçeneği değerlendirilmelidir.

---

## 3. Vekilden istenenler

**Ortak (her iki dosya):**
- [ ] Başvuru formu alanlarını resmi forma aktarma (icatçı, sahip, vekil, adres, uyrukluk, T.C. Kimlik)
- [ ] İcatçı → Sahip devri (varsa) + vekâletname
- [ ] Resmi prior-art taraması (Espacenet / Google Patents / TÜRKPATENT / WIPO) + atıf genişletme
- [ ] IPC resmî ataması
- [ ] Çizimleri PDF/TIFF (siyah-beyaz, referanslı, ayrı sayfa) olarak derleme — kaynak SVG'ler sağlanıyor
- [ ] Başvuru tarihi (öncelik) alınana kadar gizliliğin korunması

**Dosya A'ya özel:** İddia 1 (a)–(c) çekirdeğinin Dosya B ile çakışma/teklik analizi (bkz. §2).

**Dosya B'ye özel:**
- [ ] PCT/RO/101 formu doldurma + Receiving Office (TÜRKPATENT) + atanacak devletler
- [ ] IIS (uluslararası arama) + 18 ay yayım + (isteğe bağlı) 22 ay ön inceleme
- [ ] Teklik itirazına karşı divisional stratejisi (bkz. §2)

---

## 4. Dürüst teknik notlar (vekil için)

- **Doğruluk üstünlüğü iddia edilmez.** İcadın kazanımı temsil boyutu, bellek ve **native açıklanabilirlik** (aynı buluttan ablation/prototip/drift) üzerinde tanımlıdır. Fraud'da matched PR-AUC 0.462, XGBoost 0.8047'dir (üstünlük yok).
- **6.45/2.92 (dil) ölçüm sınırlaması:** v2.3'te "ortak held-out" diye etiketlense de güncel `RESEARCH.md` 6.45'in eğitim kümesi örneklemi olduğunu kaydeder; metin bunu held-out kanıtı olarak ileri sürmemektedir. Başvuru öncesi bu protokol netleştirilmelidir.
- **Düzeltmeler (C1–C4)** Dosya B'nin 7.4'ünde: permütasyon yalnız simetrik agregasyon için; fark vektörü kovaryant; paket = "kompakt/self-contained/doğrulanabilir" ilkesi (sabit KB yok); prior-art "incelenen kaynaklarda gösterilmedi" (küresel yokluk değil).

---

## 5. Ek dosya listesi

| Dosya | Tür |
|---|---|
| `docs/patent/TURKPATENT_basvuru_v2.3.md` | TR ulusal — dil (filing-ready) |
| `docs/patent/geometry_dual_use_core_v1.md` | PCT — evrensel+4 (TR) |
| `docs/patent/geometry_dual_use_core_v1_EN.md` | PCT — evrensel+4 (EN) |
| `docs/figures/sekil_1..8_*.svg` | TR dil — 8 çizim (kaynak) |
| `docs/figures/core_sekil_1..3_*.svg` | PCT — 3 yeni çizim (kaynak) |
| `aml/figures/typology_naming_*.png`, `fraud/figures/quann_cloud_3d.png`, `maintenance/figures/e1_*.png` | PCT Şekil 4-7 (dikey artifact'ler) |
| `docs/patent/patent_application_v2.3.md` + `_EN`, `docs/patent/patent_tabular_explain_v1.md` + `_EN` | Özel dosyalar (çekirdeğin alt kümeleri; cross-ref) |

> **Gizlilik (tekrar):** Bu paketin tamamı ve ekleri, ilgili başvuru tarihi (öncelik) alınana kadar ve PCT 18-ay yayımı öncesi kamuya açılmamalıdır.
