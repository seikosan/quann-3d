#!/bin/bash
# R8 hazırlık kuyruğu: QUANN pair-kanalı / head-freeze adayı.
# REVISION.md §R4: kontrol (mevcut hibrit arama) vs pair-feature yönlendirmeli
# arama ve head-freeze adayı; ölçümler Δcand + gerçek held-out PPL.
# Motor/pipeline uygulaması yoksa bu kuyruk güvenli biçimde BLOCKED ile çıkar.
cd "$(dirname "$0")/../.."  # repo root
set -euo pipefail

LOG=logs/run_r8_quann_pair.log
mkdir -p logs

echo "=== [$(date)] 1/4: MPS control — reject concurrent train/eval ===" | tee -a "$LOG"
if pgrep -af '[t]rain_|[e]val_' | tee -a "$LOG"; then
  echo "MPS busy: train_/eval_ süreci bulundu; R8 başlamadan çıkıyor." | tee -a "$LOG"
  exit 1
fi

echo "=== [$(date)] 2/4: pin code/revision state ===" | tee -a "$LOG"
git rev-parse HEAD | tee -a "$LOG"
shasum -a 256 quann_cloud/REVISION.md quann_cloud/EXPERIMENTS.md | tee -a "$LOG"

echo "=== [$(date)] 3/4: candidate design + runner gate ===" | tee -a "$LOG"
echo "R8_DESIGN | control=mevcut QUANN hibrit araması; variant=pair-feature değişimine göre arama yönü + arama sırasında head freeze, sonrasında head fine-tune; metric=Δcand ve aynı sabit TEST diliminde heldout_PPL; split=80/10/10 seed=42" | tee -a "$LOG"
echo "R8_SOURCE | pair-dominance/ko-adaptation: quann_cloud/REVISION.md:20-25; freeze ve pair-feature yönleri/test: quann_cloud/REVISION.md:61-68" | tee -a "$LOG"
echo "R8_BUDGET | EXPERIMENTS.md:46-50 first-20K/10 epoch; :54-57 expected 5-10 convergence epochs; control+variant=20 arm-epochs at 10 epochs each; wall-clock duration is not recorded there." | tee -a "$LOG"

# Bu çalıştırıcı, motor/pipeline kodu ve testleri ana oturum tarafından
# eklendikten sonra bu arayüzle sağlanmalıdır; burada model kodu değiştirilmez.
PAIR_RUNNER=${QUANN_PAIR_RUNNER:-training/quann_pair_experiment.py}
if [ ! -f "$PAIR_RUNNER" ]; then
  echo "BLOCKED: $PAIR_RUNNER yok. Gerekli kod değişikliği: mevcut QUANN eğitim hattına (1) kontrol kolu, (2) pair-feature delta'sıyla aday yönlendirme, (3) arama boyunca head freeze ve arama sonrası fine-tune, (4) seed=42 80/10/10 split ile sabit TEST heldout_PPL ve Δcand ölçümü ekleyin." | tee -a "$LOG"
  echo "Kod değişikliğinden sonra zorunlu test: python3 -m pytest quann_cloud/tests/ -v" | tee -a "$LOG"
  exit 2
fi

echo "=== [$(date)] 4/4: run matched QUANN control then pair/freeze variant ===" | tee -a "$LOG"
if python3 -u "$PAIR_RUNNER" --condition control --corpus data/turkish_sentences.txt --train-limit 20000 --epochs 10 --seed 42 --split 80/10/10 2>&1 | tee -a "$LOG"; then
  :
else
  echo "R8 control FAILED" | tee -a "$LOG"
  exit 1
fi
if python3 -u "$PAIR_RUNNER" --condition pair-freeze --corpus data/turkish_sentences.txt --train-limit 20000 --epochs 10 --seed 42 --split 80/10/10 --search-channel pair --freeze-head-during-search --finetune-head-after-search 2>&1 | tee -a "$LOG"; then
  :
else
  echo "R8 pair/freeze variant FAILED" | tee -a "$LOG"
  exit 1
fi
echo "=== [$(date)] R8 QUANN PAIR COMPLETE ===" | tee -a "$LOG"
