DURUM: vekil kararı bekliyor — gönderim nüshası değildir

# ALTERNATİF İSTEM KÜMESİ — Rule 43(2) EPC bulgusuna vekil seçeneği

Bu dosya `docs/patent/pct_v3/filing/` altındaki **ana küme metinlerini değiştirmez**; yalnız vekil
seçerse uygulanacak alternatif bir yapı önerisi olarak durur. Hukuki hüküm içermez; kararı vekil verir.

## 1. Bulgu (kaynak)

- Kayıt: `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/E_epo_claims_en.json:24-27` — §
  `term: "3 independent product claims"`; `problem: "Multiple independent claims (1, 13, 15) in the same §
  category (product). Rule 43(2) EPC generally allows only one independent claim per category."` §
  `fix: "Consider consolidating into a single independent claim per category with dependent claims for §
  alternatives, or justify under Rule 43(2)(a)-(c) EPC."` §
- Araç sayacı: aynı kayıt `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/E_epo_claims_en.json:7` → `independent_claims: 3`. §
  Ana küme sayaçları (değişmez): `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json:490` → `claim_count 15`, §
  `patent-cikti/uspto/pct_v2-uspto-inceleme/motor_2026-11-20/B_us_claims_en_antecedentON.json:491-492` → `independent_count 3 / dependent_count 12` §
- Not (olgusal): araç, yöntem (1), sistem (13) ve kayıt ortamı (15) istemlerinin üçünü de `product` §
  kategorisi olarak saymıştır; kategori sınıflandırması aracın kendi parserının çıktısıdır. §

## 2. Alternatif yapı (vekil seçerse)

- **Bağımsız istem sayısı: 1** (yalnız İstem 1 — yöntem kategorisi). Bağımlı: 14. Toplam: 15 § (sayı korunur). §
- Gerekçe: aynı kategori sayılan 13 ve 15'in bağımlıya dönüştürülmesi, bulgunun `fix` alanında önerdiği §
  "single independent claim per category" yolunun **isteam sayısını koruyan** uygulanışıdır. Bulgunun §
  ikinci yolu (Rule 43(2)(a)-(c) altında gerekçelendirme) **hiç metin değişikliği gerektirmez** — iki §
  seçenek de masadadır; hangisinin izleneceğine vekil karar verir. §
- İstemler 1-12 ve mevcut 14'ün içerik metinleri aynen kalır; yalnız mevcut 13 ve 15 aşağıdaki gibi §
  bağımlı hâle getirilir, 14 yeni 13'e bağlanır. §

## 3. Taslak bağımlı metinler (EN — vekil kararı olmadan gönderilmez)

> 13. The method of claim 1, implemented by a computer system comprising at least one processor and a §
> memory storing instructions executable by the at least one processor, wherein execution of the
> instructions configures the system to operate as: a point projection module converting the
> context-encoded representations into D-dimensional vectors to construct a point cloud consisting of
> N points, wherein the task input consists of one of an ordered sequence of tokens and a plurality of
> measured field values of a tabular record, and such that measured field values of a tabular record are
> represented as points of the point cloud of that record; a pair-geometry encoder computing point
> features for points of the point cloud and geometric pair-geometry features, dependent on positions of
> two points, for point pairs of the point cloud; an aggregation module combining the point features and
> the pair-geometry features into a single set representation by a set aggregation independent of an
> input order of the points and of the point pairs, wherein the reader produces the task output from the
> set representation; an interpretation engine producing explanation data, from the same point cloud
> from which the task output is produced, by applying at least one of a point-removal effect, region
> naming and region shift without requiring a separate explanation model; and an output interface
> jointly presenting the task output and the explanation data.

> 14. The method of claim 13, wherein the reader of the computer system carries a distribution store §
> holding, in a single self-contained package, INT8 quantized parametric layers of the reader, a §
> quantization table mapping input values to model input identities, and a resolution table mapping the
> model input identities to human-readable input labels; and wherein the interpretation engine is
> configured to compute a point-removal effect over the INT8 quantized parametric layers carried in the §
> single self-contained package.

> 15. The method of claim 1, wherein the steps of the method are stored as a set of instructions on a §
> non-transitory computer-readable recording medium, the set of instructions executable by a computer
> comprising a processor such that execution of the set of instructions causes the computer to perform
> the steps.

## 4. Uygulama notları

- TR eş kopya vekil onayından sonra bu yapıya göre yeniden türetilir ("İstem 13'e göre yöntem olup..."). §
- Bu taslakta da istemlerde referans numarası yoktur (PCT Rule 13.4; `docs/patent/pct_v3/REFERANS_NUMARALARI.md:8` kural 1). §
- Yeni sayı yok: D, N, INT8 ve tüm nitelikler mevcut `filing/CLAIMS_FILING-COPY_EN.txt` satırlarından §
  (istem 13: EN:47-57, istem 14: EN:59, istem 15: EN:61) aynen alınmıştır. §

**vekil onayı gerekli** — bu dosya başvuru nüshası değildir.
