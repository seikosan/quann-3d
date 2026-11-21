# REFACTOR PLANI — 2026-11-21

> Kaynak: tam-dizin mekanik taraması (grep/ruff/diff/difflib, oturum 2026-11-21).
> Baseline: `pytest -q` → **125 passed** (43,80 s) · `denetim_kos.py hepsi` exit 0
> (kayıt `~/.local/share/opencode/denetim/20261121-074718-hepsi.json`) ·
> `ruff --select F,E9` → 237 bulgu (F541 116 · F401 81 · F841 31 · F821 8 · F811 1) ·
> HEAD `9af3b0f`, dal `deneme_context_cloud`.

## Gerçek bug (tek)

- `training/train_wiki.py:354-356` — `step_test` içinde `texts` tanımsız. Eval bloğu
  `16a4225` (29 Ek 2026, "Ölçüm geçerliliği") ile geldi; O GÜNDEN HİÇ BAŞARIYLA
  KOŞMADI (loglarda `NameError` izi yok; 17 Ek `run_wiki1500k_posttrain.log` bloktan
  ÖNCEKI koda ait, `ADIM 4` sonrası `📊 EVALUATION` satırı basılmamış). CLI
  `--step test` bugün `eval_split>0` (default 5,0 — `:457`) ile çöker.
- Düzeltme: eval dalı içinde `texts = load_txt(args.corpus, args.max_texts)` +
  bölme mantığını `eval_train_split(texts, eval_split)` yardımcı fonksiyonuna çıkar,
  `training/tests/test_train_wiki.py`'ye regresyon testi ekle. `step_train`e dokunulmaz.
- Sahte pozitifler (aksiyon yok): `fraud/src/make_figures.py:174-178` F821 — closure
  değişkenleri `:156`'da tanımlı; `fraud/src/train_rich_points.py:166` — string
  annotation + `:168` fonksiyon-içi import.

## A — GÜVENLİ MEKANİK (uygulanıyor, davranış-nötr)

- A1: `ruff --fix --select F401,F541` aktif kodda (12 paket). Dışarıda: `docs/patent/**`,
  `patent-cikti/**`, `_archive/**` (teslimat/üretim dokunulmaz), `__init__.py` (F401 yok,
  kontrol edildi). Kapı: pytest 125 passed kalmalı.
- A2: `step_test` bug düzeltmesi + regresyon testi (yukarıda).
- A3: `README.md:136` ölü pointer: `_archive/src_legacy/visualizer/` → gerçek yol
  `_archive/refactor_2026_11/src/visualizer/` (find kanıtı).

## B — YAPISAL ADLANDIRMA (uygulanıyor)

- B1: pytest-CLI karışıklığı — `test_*.py` adlı ARAÇLAR gerçek test değil (argparse'lı
  eval/etkileşim scriptleri, testpaths dışı):
  - `contextcloud/test_long_text.py` → `eval_long_text.py` (DEVIR.md:186 pointer güncellenir)
  - `contextcloud/test_trained_model.py` → `eval_trained_model.py`
  - `purecloudlm/test_cpu_quick.py` → `eval_cpu_quick.py`
  - `training/test_model.py` → `training/eval_model.py` (README:124 + kendi docstring güncellenir)
- Kanıt: başlıkları kullanım-komutu içeriyor (`training/test_model.py:6-9`); hiçbiri
  `pytest --collect-only` (125) içinde değil.

## C — KARAR GEREKTİRİR (onaylandı 2026-11-21 — UYGULAMA: bkz. Durum bloğu)

- C1: 16 referanssız script → `_archive` adayı (`archiver` kuralı AGENTS §3.5; agent
  altyapısı şu an bozuk). Kanıt: depo içi grep 0 hit + `logs/` + `docs/` 0 hit:
  `contextcloud/{train_memory_seq,train_long_text_domain,train_hierarchical,train_long_context_baseline,test_trained_model}.py`
  (son ikisi B1 ile eval_* olur; aday kalsa da), `contextcloud/examples/*` (3),
  `purecloudlm/{test_cpu_quick,generate_samples,benchmark_performance}.py`,
  `training/{train_cloudlm_comparison,eval_r3_duel}.py`, `scripts/create_long_texts.py`,
  `maintenance/src/run_e1_cwru.py`, `fpga/tools/gen_golden.py`,
  `fraud/{scripts/download_kaggle.py,src/repro_check.py}`. NOT:
  `aml/src/e5_stability.py` logs'ta/geçmese de `docs/patent/final_v3/TEKNIK_ETKI.md`'de
  geçiyor → arşivlenmez.
- C2: `neural-atlas/neural-atlas/` İZLENEN iç-içe kopya alt ağacı (aynı asset hash'leri:
  `dist/assets/index-BgmM5QyY.js` iki kopyada) — silme/sıkıştırma kararı.
- C3: `fraud/results/run_qat_gate.py` (esik dairesel THRESHOLD `:29` civarı) eski
  sürüm; `fraud/scripts/run_qat_gate_fixed.py` düzeltmeyi belgeliyor. `results/` altında
  kod durması yanlış yerleşim; `eval_online_int8_baseline.py` ile birlikte arşiv kararı.
- C4: PATENT DOKUNULMAZ: `docs/patent/pct_v2/generate_figures_pct.py` ↔
  `pct_v3/generate_figures_pct3.py` %90,3 benzer (difflib) — başvuru üretim kodu;
  `build_docx.py:233` F811 aynı sebeple temizlenmedi.
- C5: METODOLOJİ: `step_test` "held-out"u corpus başından keser ama `step_train`
  TÜM corpus'u eğitir (`:168-172`) → AGENTS §4.1'e aykırı kıyas. Bilimsel karar
  (RESEARCH.md'ye taşınabilir), refactor kapsamı dışı.
- C6: `requirements_pretrained.txt` ayrı kalabilir; isim/netleştirme kararı.
- C7: 132 `sys.path.insert` kalıbı — çalışan konvansiyon; büyük churn, bugünkü kapsam dışı.

## Doğrulama kapısı (uygulama sonrası, sırayla)

1. `python3 -m pytest -q` → **127 passed** ✓ (125 baseline + 2 regresyon; A2 commit d944dbc)
2. `ruff check --select F,E9` aktif kodda → F401/F541/F541 = 0 ✓; kalan 33 = 28 F841
   (bilerek bırakıldı: analiz scriptlerinde niyetli olabilir, tek tek bilim kararı) +
   5 F821 sahte-pozitif ✓ (yukarıda kanıtlandı)
3. Import smoke: 12/13 ✓ — tek başarısız `maintenance.src.train_maintenance`, sebebi
   refactor değil: `import generator` bare-kalıbı `fraud/src/generator.py`'ye çarpıyor
   (pytest conftest shimi bunu paket-bazlı çözüyor; 127 passed buna dahil). Kalıbın
   kendisi C7 kapsamı.
4. `scripts/check_docs_lines.sh` OK ✓ · `denetim_kos.py hepsi --repo .` → bkz. DEVIR notu
5. Bağımsız `tester`/`reviewer` doğrulama turu (agent altyapısı çalışıyorsa; değilse
   raporda "kapı koşamadım" diye açıkça yazılır).
6. DEVIR.md `devir_yaz.py` ile tazelenir; commit'ler lokal (push yok — kullanıcı kararı).

## Durum (2026-11-21 uygulama sonrası)

- **Uygulandı A/B:** A1 (commit 78351b7) · A2 (d944dbc) · B1+A3 (3aa1eb7).
- **Uygulandı C (kullanıcı onayı 2026-11-21, "c1-c7 tamamla"):** C1 17 script →
  `_archive/refactor_2026_11_21/` (2c68ebb; commit stat'inde 18 R = 17 çalıştırılabilir
  script + `examples/__init__.py` paket dosyası, gerekçe `_archive/README.md:76`;
  `eval_trained_model.py` listeden çıkarıldı —
  KARA_KUTU_PLAN referansı; `aml/src/e5_stability.py` kaldı — TEKNIK_ETKI atfı) ·
  C2 bayat üst-düzey kopyalar silindi, tek kaynak `neural-atlas/neural-atlas/` (55649dc) ·
  C3 `run_qat_gate.py` + `eval_online_int8_baseline.py` arsiv (a531d3e;
  `run_online_cloud_separation.py` kaldı — `fraud/scripts/run_val_threshold_rerun.sh` çağırıyor) ·
  C4 KARAR: patent üretimi, DOKUNULMADI · C5 §4.1 80/10/10 uygulandı: eğitim %80,
  best seçimi val-PPL, %10 test yalnız `step_test` (`split_corpus`; checkpoint'a
  `train_ppl`/`val_ppl`/`split_seed` eklendi) · C6 `requirements_pretrained.txt`
  başlığa kapsam yazıldı (ayrı kaldı) · C7 `sys.path.insert` KEEP kararı (churn riski).
- **Kapı:** pytest 128 passed (127 + split/val regresyonları; −1 eski head-holdout testi
  `test_split_corpus_follows_agents_4_1` ile değişti) · ruff F,E9 `train_wiki*.py` temiz.

