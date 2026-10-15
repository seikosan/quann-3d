"""
QCloudTrainer — Backprop'suz Eğitim Pipeline
=============================================

QCloudNetwork kullanarak backprop'suz eğitim yapar:
- train_epoch(): Her örnek için forward pass + reward/cense ile narrownning
- evaluate_ppl(): Held-out PPL evalüasyonu (QUANN noktalarıyla)
- generate_text(): Metin üretimi (QUANN noktalarıyla)

**Backprop'suz eğitim:**
- Gradient YOK
- Credit assignment: Token-level (pozisyon-level) reward
- Search mechanism: Reward/cense + narrownning (nokta geometrisi)
- Tokenizer: Gerçek BPE (train_wiki.BPETokenizer)

**Not:** token_weights epoch'lar ARASINDA kalıcıdır — öğrenme birikir.
"""

from __future__ import annotations

import math
import time
import random
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict

try:
    from .qcloud_network import QCloudNetwork
except ImportError:
    from qcloud_network import QCloudNetwork


class QCloudTrainer:
    """
    Backprop'suz eğitim pipeline'ı.
    
    Attributes:
        model: QCloudNetwork
        tokenizer: BPETokenizer (encode/decode)
        corpus: list[str] (eğitim korpusu)
        eval_corpus: list[str] (held-out; boşsa corpus'un son %10'u)
        device: torch.device
        bidirectional: bool (karşıt yön deneme aktif mi?)
        convergence_threshold: float
        expansion_rate: float
        max_epochs: int
        eval_interval: int
        seq_len: int
        batch_size: int (eval batch)
    """
    
    def __init__(
        self,
        model: QCloudNetwork,
        corpus: List[str],
        device: torch.device,
        tokenizer=None,
        eval_corpus: Optional[List[str]] = None,
        bidirectional: bool = True,
        convergence_threshold: float = 0.01,
        expansion_rate: float = 0.1,
        max_epochs: int = 20,
        eval_interval: int = 5,
        seq_len: int = 10,
        batch_size: int = 64,
        hybrid: bool = False,
        lr: float = 2e-4,
    ):
        assert tokenizer is not None, "QCloudTrainer gerçek bir tokenizer gerektirir"
        self.model = model
        self.hybrid = hybrid
        self.lr = lr
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01) if hybrid else None
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, max_epochs)) if hybrid else None
        self.corpus = corpus
        self.eval_corpus = eval_corpus or []
        self.tokenizer = tokenizer
        self.device = device
        self.bidirectional = bidirectional
        self.convergence_threshold = convergence_threshold
        self.expansion_rate = expansion_rate
        self.max_epochs = max_epochs
        self.eval_interval = eval_interval
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device_str = device.type
        
        self._examples: Optional[List[tuple]] = None
        self._eval_examples: Optional[List[tuple]] = None
    
    # =========================================================================
    # TOKENİZASYON
    # =========================================================================
    
    def _build_examples(self, texts: List[str]) -> List[tuple]:
        """Her cümle → (ilk seq_len token, sonraki 1 token) penceresi."""
        examples = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            if len(ids) < self.seq_len + 1:
                continue
            examples.append((ids[:self.seq_len], ids[self.seq_len]))
        return examples
    
    def _get_examples(self) -> List[tuple]:
        if self._examples is None:
            t0 = time.time()
            self._examples = self._build_examples(self.corpus)
            print(f"  Tokenizasyon: {len(self._examples)} örnek ({time.time()-t0:.1f}s)")
        return self._examples
    
    def _get_eval_examples(self) -> List[tuple]:
        if self._eval_examples is None:
            if self.eval_corpus:
                self._eval_examples = self._build_examples(self.eval_corpus)
            else:
                n = max(100, len(self.corpus) // 10)
                self._eval_examples = self._build_examples(self.corpus[-n:])
        return self._eval_examples
    
    def _current_points(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mevcut QUANN noktaları → (1, T, D) tensor (token kimlikleri, padding 0)."""
        return self.model.points_for(tokens, mask)
    
    # =========================================================================
    # EĞİTİM DÖNGÜSÜ
    # =========================================================================
    
    def train_epoch(self, epoch: int, verbose: bool = True) -> Dict:
        """
        Bir epoch eğit (backprop yok, reward/cense ile narrownning).
        
        Her örnek: model.update_token_weights → aday/karşıt noktalar
        forward pass'e points_override ile verilir, hata ölçülür,
        aralıklar daraltılır/genişletilir.
        """
        t0 = time.time()
        examples = self._get_examples()
        random.shuffle(examples)
        
        if not examples:
            return {'error': 'No valid sentences'}
        
        total_rewarded = 0
        total_censured = 0
        total_unchanged = 0
        total_base_err = 0.0
        total_final_err = 0.0
        total_delta = 0.0
        n = 0
        
        eos_id = self.model.eos_id
        for tokens_ids, target_id in examples:
            if target_id == eos_id:
                continue
            tokens = torch.tensor([tokens_ids], device=self.device)
            mask = (tokens != 0) & (tokens != eos_id)
            
            if not mask[0].any():
                continue
            # Kısa cümle padding'si (seq_len'den kısa olamaz — filtrelenmiş)
            
            base_points = self._current_points(tokens, mask)
            
            # Hibrit: önce ağ backprop ile eğitilir (mevcut QUANN noktalarına),
            # sonra QUANN noktaları arama ile güncellenir
            if self.hybrid:
                self.model.train()
                logits, _, _ = self.model.forward(
                    tokens, mask, points_override=base_points)
                loss = F.cross_entropy(
                    logits, torch.tensor([target_id], device=self.device))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.model.eval()
                # (QUANN ağırlıkları numpy — backprop onları değiştirmez; base_points geçerli)
            
            search_rule = "search" if self.hybrid else "classic"
            stats = self.model.update_token_weights(
                tokens, mask, target_id, base_points, search_rule=search_rule
            )
            
            total_rewarded += stats['rewarded']
            total_censured += stats['censured']
            total_unchanged += stats['unchanged']
            total_base_err += stats.get('base_error', 0.0)
            total_final_err += stats.get('final_error', stats.get('base_error', 0.0))
            total_delta += stats.get('mean_delta', 0.0)
            n += 1
            
            if verbose and n % 2000 == 0:
                el = time.time() - t0
                print(f"    {n}/{len(examples)} örnek | "
                      f"ort. base_err={total_base_err/n:.4f} | {el:.0f}s", flush=True)
        
        elapsed = time.time() - t0
        total_trials = total_rewarded + total_censured
        result = {
            'epoch': epoch,
            'n_examples': n,
            'total_rewarded': total_rewarded,
            'total_censured': total_censured,
            'total_unchanged': total_unchanged,
            'reward_ratio': total_rewarded / total_trials if total_trials > 0 else 0.0,
            'avg_base_error': total_base_err / max(n, 1),
            'avg_final_error': total_final_err / max(n, 1),
            'avg_candidate_delta': total_delta / max(n, 1),
            'elapsed': elapsed,
        }
        
        if verbose:
            print(f"  Epoch {epoch}: R={total_rewarded}, C={total_censured}, "
                  f"U={total_unchanged}, Ratio={result['reward_ratio']:.3f}, "
                  f"base_err={result['avg_base_error']:.4f} → "
                  f"final_err={result['avg_final_error']:.4f}, "
                  f"Δcand={result['avg_candidate_delta']:+.4f}, "
                  f"Time={elapsed:.0f}s", flush=True)
        
        return result
    
    def train(self, verbose: bool = True, save_path: Optional[str] = None) -> Dict:
        """
        Tam eğitim döngüsü (backprop yok).
        
        token_weights BİR KEZ oluşturulur ve epoch'lar arasında kalıcıdır
        (öğrenme birikir, yakınsama takip edilebilir).
        """
        print(f"\n{'='*60}")
        print(f"  QUANN-Cloud Training — Backprop'suz Eğitim")
        print(f"{'='*60}")
        print(f"  Corpus: {len(self.corpus)} sentences")
        print(f"  Eval corpus: {len(self.eval_corpus) if self.eval_corpus else '(corpus son %10)'}")
        print(f"  Model: {self.model}")
        print(f"  Vocab: {self.model.vocab_size} | Device: {self.device_str}")
        print(f"  Max epochs: {self.max_epochs} | Seq len: {self.seq_len} | "
              f"Eval interval: {self.eval_interval} | "
              f"Hybrid (backprop+search): {self.hybrid}\n")
        
        # Token weights: token KİMLİĞİ bazlı, lazy oluşturulur (epoch'lar
        # arasında kalıcı — öğrenme birikir, yakınsama takip edilebilir)
        print(f"  Token weights: lazy (token kimliği bazlı, D={self.model.D})\n")
        
        best_ppl = float('inf')
        best_epoch = 0
        total_t0 = time.time()
        last_stats = None
        
        for epoch in range(1, self.max_epochs + 1):
            self.model.eval()  # dropout kapalı (deterministik arama)
            
            stats = self.train_epoch(epoch, verbose=verbose)
            
            if self.hybrid and self.scheduler is not None:
                self.scheduler.step()
            last_stats = stats
            
            if 'error' in stats:
                print(f"  Error: {stats['error']}")
                break
            
            # PPL eval
            if epoch % self.eval_interval == 0 or epoch == self.max_epochs:
                ppl = self.evaluate_ppl()
                print(f"  PPL (held-out): {ppl:.2f}", flush=True)
                
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_epoch = epoch
                    if save_path:
                        self._save(save_path, epoch, ppl)
                        print(f"  Best model saved: {save_path}", flush=True)
            
            # Yakınsama kontrolü
            if self.model.all_converged():
                print(f"  All tokens converged at epoch {epoch}!")
                break
        
        total = time.time() - total_t0
        print(f"\n  ✅ Training complete | Best PPL: {best_ppl:.2f} (epoch {best_epoch})")
        print(f"  Total time: {total:.0f}s ({total/3600:.1f}h)")
        
        return {
            'best_ppl': best_ppl,
            'best_epoch': best_epoch,
            'final_stats': last_stats,
            'total_time': total,
        }
    
    def _save(self, path: str, epoch: int, ppl: float) -> None:
        """Model + QUANN noktalarını (low/high/value/direction/sigma) kaydet."""
        ckpt = {
            'model_state': self.model.state_dict(),
            'config': {
                'vocab_size': self.model.vocab_size,
                'D': self.model.D,
                'd_model': self.model.d_model,
                'd_cloud': self.model.d_cloud,
                'n_layers': self.model.context_encoder.num_layers,
                'bidirectional': self.model.bidirectional,
            },
            'epoch': epoch,
            'ppl': ppl,
            'token_weights': [
                {
                    'token_id': tid,
                    'low': w.low.tolist(),
                    'high': w.high.tolist(),
                    'value': w.value.tolist(),
                    'direction': w.direction.tolist(),
                    'sigma': w.sigma,
                }
                for tid, w in self.model.token_weights.items()
            ],
        }
        torch.save(ckpt, path)
    
    # =========================================================================
    # DEĞERLENDİRME
    # =========================================================================
    
    def evaluate_ppl(self, n_sentences: Optional[int] = None,
                     batch: int = 64) -> float:
        """
        Held-out PPL: QUANN noktaları + gerçek BPE token'ları.
        """
        model = self.model
        model.eval()
        
        examples = self._get_eval_examples()
        if n_sentences:
            examples = examples[:n_sentences]
        if not examples:
            return float('inf')
        
        eos_id = model.eos_id
        total_loss, total_tokens = 0.0, 0
        
        with torch.no_grad():
            for b in range(0, len(examples), batch):
                chunk = examples[b:b + batch]
                tokens = torch.tensor(
                    [e[0] for e in chunk], device=self.device)
                targets = torch.tensor(
                    [e[1] for e in chunk], device=self.device)
                mask = (tokens != 0) & (tokens != eos_id)
                
                pts = model.points_for(tokens, mask)
                
                word_logits, _, _ = model.forward(tokens, mask, points_override=pts)
                # Her örnek tek next-token hedefi → 1 token/örnek
                total_loss += F.cross_entropy(word_logits, targets, reduction='sum').item()
                total_tokens += len(chunk)
        
        return math.exp(total_loss / max(total_tokens, 1))
    
    # =========================================================================
    # METİN ÜRETİMİ
    # =========================================================================
    
    def generate_text(self, prompt: str, max_new: int = 20,
                      temperature: float = 0.8) -> str:
        """Metin üretimi (gerçek tokenizer ile)."""
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.seq_len:
            ids = ids[-self.seq_len:]
        tokens = torch.tensor([ids], device=self.device)
        
        generated = self.model.generate(
            tokens, max_new=max_new, temperature=temperature
        )
        return self.tokenizer.decode(generated[0].tolist())


# =========================================================================
# TEST
# =========================================================================

if __name__ == "__main__":
    print("=== QCloudTrainer Test ===\n")
    
    # Küçük vocab'li sahte tokenizer (test için)
    class _FakeTok:
        def __init__(self):
            self.vocab = {}
            self._id = 3  # 0=pad, 1=eos, 2=unk
        def encode(self, text):
            ids = []
            for w in text.split():
                if w not in self.vocab:
                    self.vocab[w] = self._id
                    self._id += 1
                ids.append(self.vocab[w])
            return ids
        def decode(self, ids):
            inv = {v: k for k, v in self.vocab.items()}
            return ' '.join(inv.get(i, '?') for i in ids if i >= 3)
        @property
        def num_tokens(self):
            return self._id
    
    tok = _FakeTok()
    corpus = [
        "Bu bir test cümlesidir ve burada devam ediyor.",
        "Başka bir örnek cümle daha yazalım şimdi.",
        "Daha fazla veri gerekli bu küçük model için.",
        "Model yavaş yavaş öğreniyor bu görevi.",
        "Backprop yok, sadece reward ve cense var.",
    ] * 20
    
    vocab_size = 100
    model = QCloudNetwork(
        vocab_size=vocab_size,
        d_model=64,
        d_cloud=128,
        d_cloud_dim=3,
        n_layers=2,
        bidirectional=True,
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    trainer = QCloudTrainer(
        model=model,
        corpus=corpus,
        device=device,
        tokenizer=tok,
        eval_corpus=corpus[-20:],
        max_epochs=2,
        eval_interval=1,
        seq_len=5,
    )
    
    print(f"Model: {model}")
    print(f"Corpus: {len(corpus)} sentences\n")
    
    result = trainer.train(verbose=True, save_path=None)
    print(f"Best PPL: {result['best_ppl']:.2f} (epoch {result['best_epoch']})\n")
    
    generated = trainer.generate_text("Bu bir test", max_new=5)
    print(f"Generated: {generated}\n")
    
    print("=== Test Passed ===")
