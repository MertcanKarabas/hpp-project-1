# Phase 2B: Boyut İndirgeme — Seç ve Gerekçelendir

mpnet embedding matrisimiz (20.000 × 768) aramayı hızlandırmak için daha az boyuta indirgenebilir mi? Kendi power-iteration SVD'mizi yazıp kütüphane yöntemleriyle karşılaştırıyoruz ve final sistem için bir yöntem seçiyoruz.

İlgili dosyalar: [`src/svd_power.py`](../src/svd_power.py) (kendi SVD'miz), [`src/phase2b_svd.py`](../src/phase2b_svd.py) (deneyler). Ham veri: `results/phase2b_*.csv`, `results/phase2b_summary.json`.

## Özet

| Soru | Cevap |
|---|---|
| En hızlı (k=100) | PCA kovaryans çözücüsü **0,047 s**; bizim blok power iteration 0,054 s; TruncatedSVD 0,085 s; tam SVD (float32) 0,271 s |
| En yavaş | Bizim tek-vektör power iteration: k=100'de 26,6 s (ama hatası optimuma eşit) |
| Doğruluk | Hızlı yöntemlerin hepsi optimumun **%1 yakınında** |
| TruncatedSVD tam SVD'yi ne zamana kadar yener? | **k ≈ 300–350'ye kadar** (768 boyutun ~%40'ı). Bizim blok yöntemimiz k ≈ 450'ye kadar |
| PCA neden farklı? | Veri merkezli değil (ortalama vektörün uzunluğu 0,51). Cosine **merkezlenmiş koordinatlarda** hesaplanırsa sıralama değişiyor ve recall 0,74'te takılıyor. Ortalama geri eklenirse PCA, SVD ile aynı recall'ı veriyor (k=300'de 0,953 ve 0,952) |
| Seçimimiz | **Merkezlenmemiş truncated SVD**, k=300 (recall@10 = 0,95), bizim blok power iteration ile. PCA kadar iyi ama puanlaması daha basit: düz bir dot product |
| Sınav için ders | `np.linalg.svd` float32 girdiyi **gizlice float64'e çeviriyor**: 0,597 s. SciPy ile float32'de kalınca 0,271 s |

---

## 1. Fikir: SVD ve boyut indirgeme

Her matris üç parçaya ayrılabilir: **A = U · diag(σ) · Vᵀ**.
- **V'nin satırları** (sağ tekil vektörler): verideki "ana yönler", en önemliden en önemsize.
- **σ** (tekil değerler): her yönün ne kadar enerji taşıdığı.
- **U · diag(σ)**: her dokümanın bu yönlerdeki koordinatları.

Sadece ilk k yönü tutarsak her doküman 768 yerine k sayıyla temsil edilir. **Eckart–Young teoremi**: aynı rank'taki tüm matrisler arasında A'ya en yakın olanı, ilk k tekil vektörle kurulandır. Bu yüzden "en iyi mümkün hata" tam SVD'nin σ değerlerinden doğrudan hesaplanabiliyor:

$$\frac{\|A - A_k\|_F}{\|A\|_F} = \sqrt{\frac{\sum_{i>k} \sigma_i^2}{\sum_i \sigma_i^2}}$$

Karşılaştırdığımız beş yöntem:

| Yöntem | Ne yapıyor |
|---|---|
| **Bizim: power iteration** | Tekil vektörleri teker teker buluyor (ödevin "algoritmayı öğren" kısmı) |
| **Bizim: blok power iteration** | k vektörü birlikte iterate ediyor |
| **Tam SVD** (`scipy.linalg.svd`, float32) | 768 bileşenin hepsini hesaplıyor; doğruluk referansı |
| **sklearn `TruncatedSVD`** | Rastgele izdüşüm (randomized) ile sadece ilk k bileşen |
| **sklearn `PCA`** | Veriyi merkezliyor (ortalamayı çıkarıyor), sonra 768 × 768 kovaryans matrisinin özvektörlerini buluyor (`svd_solver="covariance_eigh"`) |

### 1.1 Hassasiyet tuzağı: `np.linalg.svd` float32'yi float64'e çeviriyor

NumPy'nin `linalg` modülü float32 bir matris alsa bile hesabı **float64'te** yapıyor. Matrisin float64 kopyasını çıkarıyor, LAPACK'in çift hassasiyetli rutinini (`dgesdd`) çağırıyor, sonucu float32'ye geri çeviriyor. `scipy.linalg.svd` ise float32'de kalıyor (`sgesdd`):

| Tam SVD, 20.000 × 768 | Süre |
|---|---:|
| `np.linalg.svd` (float32 girdi, içeride float64) | 0,597 s |
| `scipy.linalg.svd` (baştan sona float32) | **0,271 s** |

Diğer yöntemlerin hepsi float32 çalıştığı için adil karşılaştırma ikincisi. Aynı sebeple blok power iteration'ımızdaki QR ve küçük SVD de `scipy.linalg` kullanıyor. Bu, "dtype seçimi hızı ve belleği belirler" kuralının gerçek bir örneği.

---

## 2. Kendi SVD'miz: `svd_power.py`

### 2.1 Power iteration (tek vektör)

**Temel fikir:** Rastgele bir vektörü AᵀA ile tekrar tekrar çarparsan, en büyük tekil değere ait yön her adımda diğerlerinden daha hızlı büyür. Sonunda vektör o yöne döner. Bir yön bulununca onu çıkarıp (deflation) aynı işi sıradaki yön için tekrarlıyoruz.

```python
def power_iteration_svd(A, k, max_iter=300, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    n, d = A.shape
    V = np.zeros((d, k), dtype=A.dtype)
    S = np.zeros(k, dtype=A.dtype)
    U = np.zeros((n, k), dtype=A.dtype)
    iters = []
    for j in range(k):
        v = rng.standard_normal(d).astype(A.dtype)
        v /= np.linalg.norm(v)
        Vj = V[:, :j]
        for it in range(1, max_iter + 1):
            w = A.T @ (A @ v)
            w -= Vj @ (Vj.T @ w)
            w -= Vj @ (Vj.T @ w)
            w /= np.linalg.norm(w)
            converged = 1.0 - abs(float(w @ v)) < tol
            v = w
            if converged:
                break
        iters.append(it)
        Av = A @ v
        S[j] = np.linalg.norm(Av)
        U[:, j] = Av / S[j]
        V[:, j] = v
    return U, S, V.T, iters
```

Satır satır:
- **`v = rng.standard_normal(d)`**: Rastgele bir başlangıç vektörü. `seed=0` ile sonuç her çalıştırmada aynı.
- **`w = A.T @ (A @ v)`**: AᵀA ile çarpma. AᵀA matrisini (768 × 768) **hiç oluşturmuyoruz**; önce `A @ v` (20.000 sayı), sonra `A.T @ (...)` (768 sayı). İki matris-vektör çarpımı.
- **`w -= Vj @ (Vj.T @ w)`**: **Deflation.** Daha önce bulunan j vektörün yönlerini w'dan çıkarıyoruz (Gram–Schmidt ortogonalleştirme). Yoksa her vektör yine en büyük yöne, yani ilk vektöre yakınsardı.
- **İki kez yapılması**: float32'de tek seferlik çıkarma küçük artıklar bırakıyor. İkinci geçiş bunları temizliyor ("re-orthogonalisation").
- **`w /= np.linalg.norm(w)`**: Uzunluğu 1'e çekiyoruz; yoksa her adımda σ² ile büyüyüp taşardı.
- **`1 - |w · v| < tol`**: İki ardışık vektör aynı yönü gösteriyorsa (aralarındaki açı ~0) yakınsamış demektir.
- **`S[j] = ||A v||`, `U[:, j] = A v / σ`**: Bulunan sağ tekil vektörden tekil değeri ve sol tekil vektörü hesaplıyoruz.

**Yakınsama hızı:** j. vektör için istenmeyen yönlerin payı her iterasyonda kabaca (σⱼ₊₁/σⱼ)² oranında küçülüyor.
- İlk bileşen: σ₁ = 74,7, σ₂ = 31,7, oran² = 0,18. Sadece **5 iterasyonda** yakınsadı.
- 100. bileşen civarı: σ₁₀₀ = 5,521, σ₁₀₁ = 5,440. j = 90–110 aralığında oran² en az 0,971, ortalama **0,989**. Ortalama oranla 10⁻⁶ hassasiyete inmek için ln(10⁻⁶) / ln(0,989) ≈ **1.250 iterasyon** gerekir. Üst sınırımız 300.

k=200'de ortalama **178 iterasyon** gerekti ve **33 bileşen** 300'e takıldı (tekil değer hatası en fazla %0,95). Yine de yeniden oluşturma hatası optimumla neredeyse aynı (0,28638'e karşı 0,28637). Sebebi: tekil değerleri birbirine çok yakın olan yönler tam ayrışmasa da, yakınsamamış vektörler bu yönlerin bir karışımı ve hepsi aynı ilk-k alt uzayının içinde kalıyor. Bu yöntem matris-vektör çarpımı (BLAS-2) yaptığı için yavaş: k=100'de 26,6 s. (k=200 için tek bir çalıştırma 62 s sürdü; benchmark değil, sadece hata ve iterasyon sayısı için.)

### 2.2 Blok power iteration

```python
def block_power_svd(A, k, n_iter=6, oversample=10, seed=0):
    rng = np.random.default_rng(seed)
    n, d = A.shape
    l = min(k + oversample, d)
    V, _ = sl.qr(rng.standard_normal((d, l)).astype(A.dtype), mode="economic", check_finite=False)
    for _ in range(n_iter):
        V, _ = sl.qr(A.T @ (A @ V), mode="economic", check_finite=False)
    B = A @ V
    Ub, s, Wt = sl.svd(B, full_matrices=False, check_finite=False)
    U = Ub[:, :k]
    S = s[:k]
    Vt = (V @ Wt.T)[:, :k].T
    return U, S, Vt
```
(`sl` = `scipy.linalg`; float32 veri float32'de kalıyor.)

- **`l = k + oversample`**: k yerine k+10 vektörle çalışıyoruz. Fazladan vektörler, k. ve k+1. yön arasındaki küçük farkın yakınsamayı yavaşlatmasını engelliyor.
- **`sl.qr(...)`**: QR ayrıştırması, bloğun sütunlarını birbirine dik ve birim uzunlukta tutuyor. Tek-vektör yöntemindeki deflation'ın blok karşılığı: vektörlerin hepsinin aynı en büyük yöne çökmesini engelliyor.
- **`A.T @ (A @ V)`**: 768 × 110'luk bir blokla çarpım. Artık **matris-matris** (BLAS-3) işlem; Phase 2C'de gördüğümüz gibi çok daha verimli.
- **Rayleigh–Ritz adımı** (`B = A @ V`, küçük SVD): Döngü bize "ilk k yönü içeren alt uzayı" veriyor ama yönleri tek tek ayırmıyor. 20.000 × 110'luk küçük B matrisinin SVD'si bu alt uzayı tekil vektörlere ayırıyor.
- Bu algoritma, TruncatedSVD'nin kullandığı **randomized SVD** ile aynı fikir.

**Kaç iterasyon?** (k = 100, optimum hata = 0,42557):

| n_iter | Süre | Hata | En kötü σ hatası |
|---:|---:|---:|---:|
| 0 | 0,017 s | 0,92478 | %63 |
| 1 | 0,023 s | 0,46854 | %20 |
| 2 | 0,029 s | 0,43644 | %9,7 |
| 4 | 0,042 s | 0,42771 | %4,3 |
| **6** | **0,054 s** | **0,42630** | %2,5 |
| 8 | 0,066 s | 0,42587 | %1,5 |
| 12 | 0,091 s | 0,42564 | %0,5 |
| 16 | 0,116 s | 0,42559 | %0,15 |

n_iter = 0 rastgele bir alt uzay demek (hata 0,92). 6 iterasyonda hata optimumun sadece %0,17 üstünde. Varsayılan olarak 6'yı seçtik; sklearn'ün TruncatedSVD varsayılanına (5 iterasyon) yakın, böylece karşılaştırma adil.

---

## 3. Sonuçlar

### 3.1 Zaman vs k (20.000 × 768, saniye, 1 warmup + 5 ölçüm, median)

| k | Tam SVD (float32) | TruncatedSVD | PCA | Bizim blok | Bizim tek-vektör |
|---:|---:|---:|---:|---:|---:|
| 10 | 0,271 | 0,035 | 0,045 | **0,023** | 0,664 |
| 25 | 0,271 | 0,046 | 0,045 | **0,030** | 3,04 |
| 50 | 0,271 | 0,054 | 0,045 | **0,031** | 9,64 |
| 100 | 0,271 | 0,085 | **0,047** | 0,054 | 26,6 |
| 200 | 0,271 | 0,175 | **0,049** | 0,105 | — |
| 300 | 0,271 | 0,235 | **0,050** | 0,163 | — |
| 400 | 0,271 | 0,395 | **0,052** | 0,221 | — |
| 500 | 0,271 | 0,442 | **0,054** | 0,291 | — |
| 600 | 0,271 | 0,548 | **0,056** | 0,472 | — |
| 700 | 0,271 | 0,790 | **0,058** | 0,464 | — |

- Tam SVD k'dan bağımsız; her zaman 768 bileşenin hepsini hesaplıyor (bir kez ölçüldü).
- TruncatedSVD ve bizim blok yöntemimiz k ile kabaca doğrusal büyüyor. TruncatedSVD k = 300 ile 400 arasında, bizim blok yöntem k = 400 ile 500 arasında tam SVD'yi geçiyor.
- PCA'nın kovaryans çözücüsü k'dan neredeyse bağımsız (Bölüm 4.3).
- Tek-vektör power iteration k = 200 için 5 ölçüm ~5 dakika sürerdi; zaman ölçümü k ≤ 100 ile sınırlı.

### 3.2 Zaman vs matris boyutu (k = 100, saniye)

| Doküman | Tam SVD | TruncatedSVD | PCA | Bizim blok | Bizim tek-vektör |
|---:|---:|---:|---:|---:|---:|
| 1.000 | 0,035 | 0,011 | 0,034 | **0,007** | 0,398 |
| 5.000 | 0,082 | 0,027 | 0,037 | **0,019** | 2,89 |
| 10.000 | 0,143 | 0,047 | 0,040 | **0,030** | 13,0 |
| 20.000 | 0,398 | 0,085 | **0,047** | 0,054 | 26,4 |

- Tam SVD, TruncatedSVD ve blok yöntemi satır sayısıyla kabaca doğrusal büyüyor.
- PCA küçük matrislerde sabit ~0,035 s: bu süreye 768 × 768'lik özdeğer problemi hakim ve satır sayısına bağlı değil.
- Tek-vektör yöntemi doğrusaldan hızlı büyüyor. Matris büyüdükçe tekil değerler birbirine yaklaşıyor ve gereken iterasyon sayısı artıyor.
- Not: 20K'daki tam SVD bu tabloda 0,398 s, Bölüm 3.1'de 0,271 s. Aynı hesap, aynı çalıştırmanın iki farklı bölümünde ölçüldü. Bu LAPACK çağrısı için çalıştırmalar arası oynama ±%40'a kadar çıkabiliyor; karşılaştırmaları aynı tablonun içinde yapıyoruz.

### 3.3 Yeniden oluşturma hatası (‖A − A_k‖_F / ‖A‖_F)

| k | Optimum (Eckart–Young) | Tam SVD | TruncatedSVD | PCA | Bizim blok | Bizim tek-vektör |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0,53447 | 0,53447 | 0,53481 | 0,53323 | 0,53485 | 0,53447 |
| 100 | 0,42557 | 0,42557 | 0,42640 | 0,42492 | 0,42630 | 0,42557 |
| 200 | 0,28637 | 0,28637 | 0,28715 | 0,28607 | 0,28711 | 0,28638 |

Yorum:
- k=100'de hata 0,4256, yani enerjinin 1 − 0,4256² = **%81,9**'u korunuyor. k=50'de %71,4, k=200'de %91,8.
- Tam SVD ve tek-vektör yöntemi optimuma eşit. TruncatedSVD ve blok yöntemi optimumun %0,06–0,3 üstünde; birkaç iterasyonla durdukları için kabul edilebilir bir yaklaşıklık.
- **PCA optimumun altında görünüyor.** Bu bir çelişki değil: PCA k bileşene ek olarak ortalama vektörünü de saklıyor, yani aslında rank k+1 bir yaklaşım yapıyor.

### 3.4 Bellek (tracemalloc ile tepe bellek, k = 100)

| Yöntem | Tepe bellek | Neden |
|---|---:|---|
| TruncatedSVD | 79 MB | Rastgele izdüşüm ve ara matrisler |
| **Tam SVD** (float32) | 64 MB | 768 bileşenin hepsi (U: 20.000 × 768 = 61 MB) |
| PCA (kovaryans) | 26 MB | Sadece 768 × 768 kovaryans + çıktı; veriyi kopyalayarak merkezlemiyor |
| **Bizim blok** | 18 MB | 20.000 × 110'luk ara matrisler, küçük SVD |
| Bizim tek-vektör | **8,5 MB** | Sadece U (20.000 × 100) ve birkaç vektör |

Veri matrisinin kendisi 61 MB. Aynı tam SVD `np.linalg` ile (float64 içeride) ilk ölçümümüzde **520 MB** tutmuştu: float64 kopya, float64 U ve float64 çalışma alanı. Hassasiyet tuzağı (Bölüm 1.1) belleği de 8 kat büyütüyor.

### 3.5 Arama kalitesi: boyut indirgeme aramayı ne kadar bozuyor?

Phase 1'deki 768 boyutlu naive aramanın top-10 sonuçlarını referans aldık. İndirgenmiş uzayda yapılan aramanın bu 10 sonuçtan kaçını bulduğunu (**recall@10**) ölçtük. PCA'yı iki şekilde puanladık:

```python
# 1) SVD ve "merkezlenmiş PCA": vektörleri k boyuta izdüşür, normalize et, dot product
Z  = normalise((A  - m) @ Vt.T)     # m = 0 for SVD, data mean for PCA
Qz = normalise((Qm - m) @ Vt.T)
S  = Qz @ Z.T

# 2) "Ortalama geri eklenmiş PCA": orijinal query ile geri kurulmuş doküman (m + z V) arasındaki cosine,
#    yine k boyutta hesaplanır:  (q·m + (qVᵀ)·z) / ||m + zV||
Zc    = (A - m) @ Vt.T
norms = np.sqrt(m @ m + 2 * Zc @ (m @ Vt.T) + (Zc * Zc).sum(1))
S     = (normalise(Qm) @ Vt.T) @ (Zc / norms[:, None]).T + (normalise(Qm) @ m)[:, None] / norms
```

| k | TruncatedSVD (merkezlenmemiş) | PCA, merkezlenmiş cosine | PCA, ortalama geri eklenmiş | Çarpım süresi (SVD) | İndirgenmiş corpus |
|---:|---:|---:|---:|---:|---:|
| 25 | 0,329 | 0,266 | 0,330 | 0,58 ms | 2 MB |
| 50 | 0,535 | 0,435 | 0,539 | 0,68 ms | 4 MB |
| 100 | 0,724 | 0,586 | 0,729 | 0,86 ms | 8 MB |
| 200 | 0,882 | 0,700 | 0,884 | 1,25 ms | 16 MB |
| **300** | **0,952** | 0,727 | **0,953** | **1,87 ms** | **24 MB** |
| 400 | 0,974 | 0,729 | 0,972 | 2,39 ms | 32 MB |
| 600 | 0,994 | 0,738 | 0,994 | 3,52 ms | 48 MB |
| 768 (tam) | 1,000 | — | — | 4,64 ms | 61 MB |

- Merkezlenmemiş SVD ve ortalaması geri eklenmiş PCA **aynı** recall'ı veriyor; ikisi de k arttıkça 1'e yaklaşıyor.
- Cosine merkezlenmiş PCA koordinatlarında hesaplanırsa recall ~0,74'te takılıyor; sebebi Bölüm 4.2'de.
- Ortalamayı geri eklemenin bedeli, düz çarpıma ek bir düzeltme terimi: k=300'de 2,44 ms (SVD 1,87 ms).
- k = 300'de recall 0,95, matris çarpımı 768 boyuta göre 2,5 kat hızlı (1,87'ye karşı 4,64 ms), bellek 2,6 kat az.

![Figure 3](../figures/fig3_svd.png)

> **Figure 3:** (a) İlk 100 bileşeni bizim blok power iteration 5,0 kat, TruncatedSVD 3,2 kat daha hızlı buluyor (float32 tam SVD'ye göre). Bu avantaj TruncatedSVD için k ≈ 300–350'de, bizim yöntem için k ≈ 450'de kayboluyor. PCA'nın kovaryans çözücüsü bu "uzun ve dar" matriste her k'da en hızlısı. (c) Hızlı yöntemlerin hepsi optimum hatanın %1 yakınında. (d) Merkezlenmemiş SVD ve ortalaması geri eklenmiş PCA k=300'de recall 0,95'e ulaşıyor; cosine'i merkezlenmiş PCA koordinatlarında hesaplamak ise recall'ı 0,74'te tutuyor.

---

## 4. Ödevin soruları

### 4.1 TruncatedSVD neden tam SVD'den hızlı?
- **Tam SVD** her zaman 768 bileşenin hepsini hesaplıyor: maliyeti O(N·d²). LAPACK'in kullandığı ilk adım (bidiagonalizasyon) kısmen matris-vektör işlemlerinden oluşuyor ve bellek hızıyla sınırlı.
- **TruncatedSVD** sadece k+10 boyutlu bir alt uzay arıyor. Rastgele bir izdüşüm yapıyor, birkaç power iterasyonu ve küçük bir SVD ekliyor. Bunların hepsi matris-matris çarpımı (BLAS-3) ve maliyet O(N·d·k).
- k = 100'de bu fark TruncatedSVD için **3,2 kat** (0,085 s'ye karşı 0,271 s), bizim blok yöntem için **5,0 kat** (0,054 s).

### 4.2 PCA SVD'den ne zaman farklıdır?
PCA, **merkezlenmiş** verinin SVD'si: önce her sütundan ortalamayı çıkarıyor. Verinin ortalaması sıfırsa ikisi aynı sonucu verir. Bizim verimizde değil:
- Ortalama vektörün uzunluğu **0,51**. Satırlar birim uzunlukta olduğu için bu çok büyük: tüm embedding'ler ortak bir yöne doğru yığılmış. Sentence embedding'lerin bilinen bir özelliği (anizotropi).
- SVD'nin ilk bileşeni bu ortalama yönüyle **neredeyse aynı** (cosine = 0,998) ve tüm enerjinin **%28'ini** taşıyor. İkinci bileşen sadece %5.
- PCA bu yönü ayrı bir vektör (ortalama) olarak saklıyor ve k bileşenini ortalama etrafındaki farklara harcıyor. Bu yüzden yeniden oluşturmada biraz daha iyi: ortalama, bedava bir ek bileşen gibi.
- **Arama için önemli olan, cosine'in nerede hesaplandığı.** Merkezlenmiş koordinatlarda iki vektörün açısı, ortalamadan *sapmalarının* açısı. Bu, orijinal cosine ile aynı sıralamayı vermiyor ve recall k = 600'de bile 0,74. Ortalamayı geri ekleyip cosine'i orijinal uzayda hesaplarsak (Bölüm 3.5'teki formül), PCA'nın alt uzayı SVD'ninki kadar iyi: k = 300'de 0,953 ve 0,952.
- Sparse TF-IDF için ek bir sorun: merkezleme sparsity'yi yok eder (her sıfır −ortalama olur) ve 2 GB'lık dense bir matris oluşur. Bu yüzden sparse veride TruncatedSVD (LSA) kullanılır.

### 4.3 TruncatedSVD'nin hâlâ kazandığı pratik maksimum k nedir?
- **Tam SVD'ye (float32) karşı:** k ≈ **300–350**. k = 300'de 0,235 s'ye karşı 0,271 s ile kazanıyor; k = 400'de 0,395 s'ye karşı 0,271 s ile kaybediyor. Yani d = 768'in yaklaşık **%40'ı**. Bizim blok yöntemimiz daha az ek iş yaptığı için k ≈ 450'ye kadar kazanıyor.
- **Kovaryans/Gram yaklaşımına karşı** (PCA'nın kullandığı yöntem): bu matriste **hiçbir k'da kazanamıyor**. AᵀA'yı (768 × 768) bir kez hesaplamak tek bir büyük matris çarpımı ve çok hızlı. 768 × 768'lik özdeğer problemi de küçük. Bu hile sadece **d küçükken** işe yarıyor. TF-IDF'te d = 25.933 olurdu: 2,7 GB'lık bir Gram matrisi ve O(d³) özdeğer problemi. Orada tek pratik seçenek TruncatedSVD.

### 4.4 Final sistem için hangi yöntem ve neden?

**Seçim: merkezlenmemiş truncated SVD, k = 300.** Tabanı bizim blok power iteration ile hesaplıyoruz (n_iter = 6); sklearn TruncatedSVD ile çapraz kontrol ediyoruz.

Gerekçeler:
1. **Merkezlenmemiş SVD veya ortalaması geri eklenmiş PCA**, ikisi de recall'da eşit (k=300'de 0,952 ve 0,953). SVD'yi seçmemizin sebebi sadelik: puan düz bir dot product. PCA'nın doğru kullanımı her puana bir düzeltme terimi ve doküman başına bir normalizasyon ekliyor (2,44 ms'ye karşı 1,87 ms). Cosine'i merkezlenmiş PCA koordinatlarında hesaplamak ise recall'ı 0,74'te tavanlıyor.
2. **k = 300:** recall@10 = 0,95 (aranan 10 dokümanın ortalama 9,5'i bulunuyor), matris çarpımı 2,5 kat hızlı, corpus belleği 61 MB'tan 24 MB'a iniyor. k = 200 (0,88) fazla kayıplı; k = 400 (0,97) ek %2 recall için %28 daha uzun matris çarpımı demek.
3. **Tabanı hangi yöntemle hesapladığımız ikincil:** Taban bir kez, offline hesaplanıyor (k = 300'de bizim yöntem 0,16 s, TruncatedSVD 0,24 s; embedding üretmek 17 dakikaydı). Kendi yöntemimiz bu boyutta en hızlı truncated yöntem, aynı doğrulukta ve her satırını açıklayabiliyoruz.
4. **Tam SVD gereksiz:** 768 bileşenin hepsini hesaplıyor, biz sadece 300'ünü kullanacağız.
5. **Kovaryans/Gram yöntemi en hızlısı olurdu** (merkezlemeden yapılırsa `eigh(AᵀA)`). Fit süresi zaten önemsiz olduğu için tercihimizi değiştirmiyor.

**Dürüst bir uyarı:** 20.000 dokümanda 768 boyutlu arama zaten hızlı (200 query için ~5 ms). Boyut indirgemenin bu ölçekteki kazancı mütevazı (Phase 4: toplu aramada 1,9 kat, tek query'de 2,4 kat) ve bir doğruluk bedeli var: tam 768 boyutlu arama baseline'la 10⁻⁵ toleransta eşleşir, indirgenmiş arama eşleşmez. Bu yüzden final sistem iki mod sunuyor: **tam mod** (768 boyut, baseline ile birebir) ve **hızlı mod** (k = 300, recall 0,95). İndirgeme asıl değerini 10 kat büyük bir corpus'ta gösterir: 200.000 dokümanda 614 MB yerine 240 MB okumak.

---

## 5. Düzeltme notu

İlk sürümde iki hata vardı. Proje baştan sona gözden geçirilirken bulundu ve deney yeniden çalıştırıldı:

1. **Tam SVD için `np.linalg.svd` kullanılmıştı.** Bu fonksiyon float32 girdiyi float64'te hesaplıyor (Bölüm 1.1). Tam SVD bu yüzden olduğundan yavaş (0,62 s) ve bellek dolu (520 MB) görünüyordu. Sonuç olarak "TruncatedSVD 6,7 kat hızlı" ve "k ≈ 550'ye kadar kazanıyor" sonuçları abartılıydı. Doğruları 3,2 kat ve k ≈ 300–350.
2. **PCA'nın recall'ı yalnızca merkezlenmiş cosine ile ölçülmüştü,** ve "PCA aramada kötü" sonucuna varılmıştı. Ortalama geri eklenince PCA, SVD kadar iyi. Fark PCA'nın kendisinde değil, puanın nasıl hesaplandığında.

Ayrıca blok power iteration'ın QR ve SVD adımları `scipy.linalg`'e geçirildi (float32'de kalsın diye). Bu, yöntemimizi k=100'de 0,079 s'den 0,054 s'ye, belleğini 80 MB'tan 18 MB'a indirdi.

---

## 6. Yeniden üretmek için

```bash
.venv/bin/python src/phase2b_svd.py              # ~11 dk (tek-vektör power iteration'ın payı büyük)
.venv/bin/python src/phase2b_svd.py --plot-only  # sadece figür
```

---

## 7. Sunumda gelebilecek sorular

- **Power iteration neden çalışıyor?** AᵀA ile her çarpımda her yön σ² ile büyüyor. En büyük σ'ya ait yön diğerlerini geride bırakıyor ve vektör o yöne dönüyor.
- **AᵀA'yı neden hiç oluşturmadınız?** `A.T @ (A @ v)` iki matris-vektör çarpımıyla aynı sonucu veriyor ve 768 × 768'lik ara matrise gerek kalmıyor. Büyük d'de (TF-IDF: 25.933) bu matris 2,7 GB olurdu.
- **Deflation ne işe yarıyor?** Bulunan yönleri sonraki vektörden çıkarıyor. Yoksa her vektör yine ilk (en büyük) yöne yakınsardı.
- **Neden bazı bileşenler 300 iterasyonda yakınsamadı?** Yakınsama hızı (σⱼ₊₁/σⱼ)². 100. bileşen civarında bu oran ortalama 0,989; her iterasyonda istenmeyen yönün sadece ~%1'i kayboluyor ve 10⁻⁶'ya inmek ~1.250 iterasyon ister.
- **Yakınsamadıysa hata neden optimum?** Tekil değerleri çok yakın yönler aynı alt uzayın içinde. Vektör hangi karışımda kalırsa kalsın, k-boyutlu uzay neredeyse aynı.
- **Blok yöntemi neden ~490 kat hızlı?** Tüm vektörleri birlikte işliyor: matris-vektör yerine matris-matris çarpımı (BLAS-3). Ayrıca tek tek yönleri değil alt uzayı arıyor; yakın tekil değerler onu yavaşlatmıyor.
- **`np.linalg.svd` neden yavaştı?** float32 girdiyi float64'e çevirip hesaplıyor. `scipy.linalg.svd` float32'de kalıyor: 0,60 s yerine 0,27 s ve 8 kat daha az bellek.
- **PCA mı SVD mi?** Recall'da eşitler, ama PCA'yı doğru kullanmak için ortalamayı geri eklemek gerekiyor. Merkezlenmiş koordinatlarda cosine almak sıralamayı değiştiriyor (recall 0,74). SVD'yi sadelik için seçtik.
- **Neden k = 300?** Recall 0,95 ile hız (2,5 kat) ve bellek (2,6 kat) arasında iyi bir denge. Daha düşük k'larda recall hızla düşüyor, daha yüksek k'larda kazanç azalıyor.
- **TruncatedSVD neden k = 400'de tam SVD'den yavaş?** Maliyeti k ile doğrusal büyüyor; tam SVD'nin maliyeti sabit. k, d'ye yaklaşınca "sadece bir kısmını hesapla" avantajı kalmıyor.
