"""
QCloudWeight — Her Token'ın Noktası (D Boyutlu)
================================================

Mevcut QWeight (tek boyutlu) → QCloudWeight (D boyutlu).
Her token'ın buluttaki noktası bir QCloudWeight nesnesidir.

Her boyut için bağımsız narrownning yapılır:
  - low[i]: i'inci boyut için aralık alt sınırı (başlangıç: -1.0)
  - high[i]: i'inci boyut için aralık üst sınırı (başlangıç: 1.0)
  - value[i]: i'inci boyuttaki anlık nokta konumu (başlangıç: random[-1, 1])
  - direction[i]: i'inci boyuttaki öğrenme yönü (+1/-1)
  - sigma: Arama belirsizliği (başlangıç: 0.25)
  - outcomes: Son başarı/başarısızlık penceresi

Öğrenme = Yakınsama: Aralık bir eşik altına düştüğünde token "öğrenmiş" sayılır.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class QCloudWeight:
    """
    D Boyutlu Cloud Weight — Her token'ın buluttaki noktası.
    
    Attributes:
        D: int  # Boyut sayısı (başlangıç: 16)
        low: np.array[D]  # Aralık alt sınırı (başlangıç: -1.0)
        high: np.array[D]  # Aralık üst sınırı (başlangıç: 1.0)
        value: np.array[D]  # Şu anki nokta konumu (başlangıç: random[-1, 1])
        direction: np.array[D]  # Her boyut için yön (+1/-1)
        sigma: float  # Arama belirsizliği (başlangıç: 0.25)
        outcomes: list[bool]  # Son başarı/başarısızlık penceresi
        history: list[tuple]  # Path kaydı (value, direction, 'reward/censure')
        convergence_threshold: float  # Yakınsama eşik değeri (başlangıç: 0.01)
        expansion_rate: float  # Aralık genişleme oranı (başlangıç: 0.1)
    """
    
    D: int = 16  # Boyut sayısı
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    value: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    sigma: float = 0.25
    outcomes: List[bool] = field(default_factory=list)
    history: List[Tuple[np.ndarray, int, str]] = field(default_factory=list)
    convergence_threshold: float = 0.01
    expansion_rate: float = 0.1
    
    def __post_init__(self):
        """İlk başlatma: value ve direction rastgele başlatılır."""
        if self.low is None:
            self.low = np.full(self.D, -1.0)
        if self.high is None:
            self.high = np.full(self.D, 1.0)
        if self.value is None:
            self.value = np.random.uniform(self.low, self.high)
        if self.direction is None:
            self.direction = np.ones(self.D, dtype=int)  # Her boyut +1
        
        # Validasyon
        if not np.all(self.low < self.high):
            raise ValueError(
                f"QCloudWeight low < high olmalı. Alındı: [{self.low}, {self.high}]"
            )
    
    # =========================================================================
    # TEMEL İŞLEMLER
    # =========================================================================
    
    def sample(self) -> np.ndarray:
        """
        Yön ve sigma'ya göre aralık içinde aday üret.
        
        Return:
            np.array[D]: Yeni üretilen nokta (value değişmez)
        """
        width = self.high - self.low
        step = np.random.uniform(0.0, self.sigma, size=self.D) * width
        candidate = self.value + self.direction * step
        return np.clip(candidate, self.low, self.high)
    
    def set_value(self, new_value: np.ndarray) -> None:
        """Value'yu doğrudan belirle (testler ve iç kullanım için)."""
        if new_value.shape != (self.D,):
            raise ValueError(f"new_value shape {(new_value.shape,)} != ({self.D,})")
        self.value = new_value.copy()
    
    def narrow(self, new_point: np.ndarray, reward: bool) -> None:
        """
        Her boyut için bağımsız narrownning.
        
        Args:
            new_point: np.array[D] — Yeni denenen nokta konumu
            reward: bool — True → ödül (doğru yönde), False → ceza (yanlış yönde)
        
        Raises:
            ValueError: new_point shape veya mantıksal hata
        """
        if new_point.shape != (self.D,):
            raise ValueError(f"new_point shape {(new_point.shape,)} != ({self.D,})")
        
        for i in range(self.D):
            # Validasyon: dışarıda kalırsa aralığı o değere kadar genişlet
            if new_point[i] < self.low[i]:
                self.low[i] = new_point[i]
            elif new_point[i] > self.high[i]:
                self.high[i] = new_point[i]
            
            if reward:
                # ÖDÜL: Doğru yönde → aralığı genişlet
                # Yeni değer yüksekse, üst sınırı genişlet
                if new_point[i] > self.value[i]:
                    self.high[i] = min(1.0, self.high[i] + self.expansion_rate)
                # Yeni değer düşükse, alt sınırı genişlet
                elif new_point[i] < self.value[i]:
                    self.low[i] = max(-1.0, self.low[i] - self.expansion_rate)
                # Eşitse, dokunma
            else:
                # CEZA: Yanlış yönde → aralığı daralt
                # Eğer yeni değer eskisinden büyükse, alt sınırı yukarı çek
                if new_point[i] > self.value[i]:
                    self.low[i] = new_point[i]
                # Eğer yeni değer eskisinden küçükse, üst sınırı aşağı çek
                elif new_point[i] < self.value[i]:
                    self.high[i] = new_point[i]
                # Eşitse, dokunma
            
            # Yonu güncelle (eğer ödülse aynı yönde, ceza ise ters)
            if reward:
                if new_point[i] > self.value[i]:
                    self.direction[i] = 1
                elif new_point[i] < self.value[i]:
                    self.direction[i] = -1
            else:
                self.direction[i] = -self.direction[i]  # Yön ters çevrilir
        
        # History'ye kaydet
        self.history.append((new_point.copy(), self.direction.copy(), 
                            'reward' if reward else 'censure'))
        
        # Value'yü güncelle
        self.value = new_point.copy()
    
    def narrow_search(self, new_point: np.ndarray, reward: bool) -> None:
        """
        Düzeltmeli arama semantiği (hibrit mod için).
        
        Klasik adaptif arama kuralları:
        - ÖDÜL (iyi deneme): value = new_point; aralık, eski değer ile yeni
          değer ARASINDAKİ segmente daraltılır (lokalite bahsi).
        - CEZA (kötü deneme): value DEĞİŞMEZ; aralık, kötü noktanın dışını
          keser (kötü taraf eleme).
        
        Not: narrow() (orijinal v0 tasarımı) ödülde genişletir, cezada kötü
        tarafa daraltır ve value'yu her durumda yeni noktaya taşır.
        narrow_search() bunların tersini, standart adaptif arama kurallarıyla yapar.
        """
        if new_point.shape != (self.D,):
            raise ValueError(f"new_point shape {new_point.shape} != ({self.D},)")
        
        for i in range(self.D):
            v = self.value[i]
            c = new_point[i]
            if c > v:
                if reward:
                    # İyi deneme: segment [v, c]
                    self.low[i] = max(self.low[i], v)
                    self.high[i] = min(self.high[i], c)
                    self.direction[i] = 1
                else:
                    # Kötü deneme: c'nin ötesini kes, value'da kal
                    self.high[i] = min(self.high[i], c)
                    self.direction[i] = -1
            elif c < v:
                if reward:
                    # İyi deneme: segment [c, v]
                    self.high[i] = min(self.high[i], v)
                    self.low[i] = max(self.low[i], c)
                    self.direction[i] = -1
                else:
                    # Kötü deneme: c'nin altını kes, value'da kal
                    self.low[i] = max(self.low[i], c)
                    self.direction[i] = 1
            # c == v: dokunma
            
            # Güvenlik: aralık bozulmasın
            if self.low[i] >= self.high[i]:
                self.low[i] = v - 1e-3
                self.high[i] = v + 1e-3
        
        if reward:
            self.value = new_point.copy()
        # ceza: value değişmez
        
        self.history.append((new_point.copy(), self.direction.copy(),
                             'reward' if reward else 'censure'))
    
    def opposite(self, point: np.ndarray) -> np.ndarray:
        """
        Bir noktanın karşıt yönündeki nokta (düşük bölge).
        
        Args:
            point: np.array[D] — Mevcut nokta
            
        Returns:
            np.array[D] — Karşıt yöndeki nokta
        """
        if point.shape != (self.D,):
            raise ValueError(f"point shape {(point.shape,)} != ({self.D,})")
        
        width = self.high - self.low
        step = np.maximum(self.sigma * width, 1e-8)
        # Opposite is the mirror move against the active search direction.
        return np.clip(self.value - self.direction * step, self.low, self.high)
    
    # =========================================================================
    # YARDIMCI FONKSİYONLAR
    # =========================================================================
    
    def error_width(self) -> np.ndarray:
        """Mevcut her boyut için aralık genişliğini döndür."""
        return self.high - self.low
    
    def is_converged(self, threshold: Optional[float] = None) -> bool:
        """
        Aralık yeterince daraldı mı? (her boyut için ayrı)
        
        Args:
            threshold: Yakınsama eşik değeri (varsayılan: convergence_threshold)
        
        Returns:
            bool — True → her boyut threshold'un altında
        """
        thresh = self.convergence_threshold if threshold is None else threshold
        return bool(np.all(self.error_width() <= thresh))
    
    def update_sigma(self, rewarded: bool) -> None:
        """
        Sigma'yı başarı penceresinden uyarla (1/5 kuralının pencere kuralı).
        
        Args:
            rewarded: bool — Bu deneme başarılı mı?
        """
        self.outcomes.append(rewarded)
        if len(self.outcomes) > 10:  # Pencere boyutu
            self.outcomes.pop(0)
        
        if len(self.outcomes) >= 5:
            rate = sum(self.outcomes) / len(self.outcomes)
            shrink = 0.82
            target = 0.2
            floor = 0.02
            cap = 0.5
            
            if rate > target:
                self.sigma = self.sigma * shrink
            else:
                self.sigma = self.sigma / shrink
            self.sigma = min(cap, max(floor, self.sigma))
    
    def info(self) -> dict:
        """Ağırlık hakkında özet bilgi döndür."""
        return {
            'D': self.D,
            'low': self.low.tolist(),
            'high': self.high.tolist(),
            'value': self.value.tolist(),
            'direction': self.direction.tolist(),
            'width': self.error_width().tolist(),
            'converged': self.is_converged(),
            'sigma': self.sigma,
            'history_len': len(self.history),
        }
    
    def reset(self) -> None:
        """Ağırlığı sıfırla (yeni eğitim için)."""
        self.low = np.full(self.D, -1.0)
        self.high = np.full(self.D, 1.0)
        self.value = np.random.uniform(self.low, self.high)
        self.direction = np.ones(self.D, dtype=int)
        self.outcomes.clear()
        self.history.clear()
        self.sigma = 0.25
    
    def __repr__(self) -> str:
        status = "✓ converged" if self.is_converged() else \
                 f"width={self.error_width().mean():.4f}"
        return f"QCloudWeight(D={self.D}, val={self.value.mean():.3f}, {status})"


# =========================================================================
# TEST
# =========================================================================

if __name__ == "__main__":
    print("=== QCloudWeight Test ===\n")
    
    # Basit test: D=3
    w = QCloudWeight(D=3)
    print(f"Başlangıç: {w}")
    print(f"low: {w.low}")
    print(f"high: {w.high}")
    print(f"value: {w.value}")
    print(f"direction: {w.direction}\n")
    
    # Sample test
    sample = w.sample()
    print(f"Sample: {sample}")
    
    # Narrow test (reward)
    print("\n--- Reward Test ---")
    w.narrow(sample, reward=True)
    print(f"After reward: {w}")
    print(f"low: {w.low}")
    print(f"high: {w.high}")
    print(f"value: {w.value}")
    print(f"direction: {w.direction}")
    
    # Narrow test (censure)
    print("\n--- Censure Test ---")
    sample2 = np.array([0.5, -0.5, 0.5])
    w.narrow(sample2, reward=False)
    print(f"After censure: {w}")
    print(f"low: {w.low}")
    print(f"high: {w.high}")
    print(f"value: {w.value}")
    print(f"direction: {w.direction}")
    
    # Opposite test
    print("\n--- Opposite Test ---")
    opp = w.opposite(sample)
    print(f"Opposite of {sample}: {opp}")
    
    # Sigma adaptation
    print("\n--- Sigma Adaptation ---")
    for i in range(10):
        w.update_sigma(rewarded=(i % 3 == 0))  # %33 başarı
    print(f"Sigma after 10 steps: {w.sigma}")
    print(f"Outcomes: {w.outcomes}")
    
    # Convergence test
    print("\n--- Convergence Test ---")
    for _ in range(20):
        sample = w.sample()
        w.narrow(sample, reward=True)  # Sürekli ödül
        if w.is_converged():
            print(f"Converged after {len(w.history)} steps!")
            break
    else:
        print(f"Not converged after 20 steps. Width: {w.error_width()}")
    
    print("\n=== Test Passed ===")
