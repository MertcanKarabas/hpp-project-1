# Phase 4: Final Sistem ve Ölçeklenme

Önceki fazlardaki tüm kararları tek bir arama motorunda birleştirdik. Ardından doğruluğunu baseline'a karşı kontrol ettik, 1K'dan 20K'ya (ve sentetik olarak 200K'ya) kadar ölçeklenmesini ölçtük ve canlı bir demo hazırladık.

İlgili dosyalar: [`src/search_engine.py`](../src/search_engine.py) (final sistem), [`src/phase4_final.py`](../src/phase4_final.py) (deneyler), [`demo.py`](../demo.py) (canlı demo), [`src/extra_measurements.py`](../src/extra_measurements.py) (encode süresi). Ham veri: `results/phase4_*.{csv,json}`, `results/extra_encode_latency.json`.

## Özet

| Soru | Cevap |
|---|---|
| Final süre (200 query × 20K doküman) | **Tam mod 4,84 ms**, hızlı mod (SVD k=300) 2,51 ms. Farklı çalıştırmalarda tam mod 4,5–5,0 ms arasında |
| Naive baseline'a göre | Aynı çalıştırmada ölçülen naive süreye (154,5 s) göre tam mod **~32.000 kat**, hızlı mod ~62.000 kat |
| Doğru mu? | **Evet.** 200 query'nin 200'ünde top-10 listesi baseline ile birebir aynı (sıra dahil). Skor farkı en fazla 8,4 × 10⁻⁷ (tolerans 10⁻⁵) |
| Doğrusal ölçekleniyor mu? | Toplu aramada **evet**: 5K'dan 200K'ya kadar 1.000 doküman başına ~230–290 µs |
| Nerede bozuluyor? | Tek query'de, corpus 16 MB'lık L2 önbelleğin boyutunu aşınca: 5K → 10K arası süre 2 kat değil 3,7 kat artıyor. Bellekte de skor matrisi N ile büyüyor; parçalı arama bunu 10 kat azaltıyor |
| Canlı demo | Tek query aramada 0,87 ms (benchmark), cümleyi vektöre çevirmek CPU'da ~15–19 ms |

---

## 1. Final sistem: `search_engine.py`

```
veri → embedding (mpnet, offline) → [isteğe bağlı SVD] → skorlama (tek BLAS çarpımı) → paralel top-k
```

Her parçanın gerekçesi bir ölçüme dayanıyor:

| Karar | Kaynak |
|---|---|
| Dense mpnet vektörleri, kurulumda bir kez normalize: cosine = dot product, query başına norm hesabı yok | Phase 1: norm hesabı zamanın %58'iydi. Phase 2A: anlamsal kalite |
| Bir batch'teki tüm query'ler tek bir matris çarpımıyla (`Q @ X.T`) | Phase 2C: NumPy/BLAS en hızlısı. Phase 3: ek thread'ler matris çarpımını hızlandırmıyor |
| Top-k, kendi tek geçişli Numba kernel'imizle, query'ler üzerinde paralel | Phase 2C: NumPy argpartition'dan 98 kat hızlı |
| İsteğe bağlı hızlı mod: merkezlenmemiş SVD, 300 boyut, bizim blok power iteration ile | Phase 2B: recall@10 ≈ 0,95; PCA ile eşit ama puanlaması daha basit |
| İsteğe bağlı parçalı arama | Phase 3: 10 kat büyük corpus sorusu (bellek) |

```python
class SearchEngine:
    def __init__(self, corpus, mode="exact", dims=300, chunk_docs=None, svd_iters=6):
        ...
        X = normalise(corpus)
        self.P = None
        if mode == "fast":
            _, _, Vt = block_power_svd(X, dims, n_iter=svd_iters)
            self.P = np.ascontiguousarray(Vt.T)                 # 768 x dims, bir kez hesaplanır
            X = normalise(X @ self.P)
        self.X = np.ascontiguousarray(X, dtype=np.float32)

    def prepare(self, Q):
        Q = normalise(np.atleast_2d(Q))
        if self.P is not None:
            Q = normalise(Q @ self.P)
        return np.ascontiguousarray(Q)

    def search(self, Q, k=10):
        if not 1 <= k <= self.X.shape[0]:
            raise ValueError(...)
        Q = self.prepare(Q)
        if self.chunk_docs is None:
            return topk(Q @ self.X.T, k)
        parts = []
        for a in range(0, self.X.shape[0], self.chunk_docs):
            idx, val = topk(Q @ self.X[a:a + self.chunk_docs].T, k)
            parts.append((idx + a, val))
        return K.merge_topk(parts, k)
```

Satır satır:
- **`normalise(corpus)`**: Her doküman vektörü bir kez, kurulumda birim uzunluğa getiriliyor. Böylece aramada cosine'in paydası her zaman 1 ve iş düz bir dot product'a iniyor. Phase 1'in en büyük bottleneck'i (%58) tamamen ortadan kalkıyor.
- **`block_power_svd(...)`** (hızlı mod): 300 boyutlu taban bir kez hesaplanıyor (20K'da 0,19 s). Corpus bu tabana izdüşürülüp tekrar normalize ediliyor.
- **`prepare(Q)`**: Query'ler de aynı uzaya getiriliyor. `np.atleast_2d` sayesinde tek bir query (768 sayılık vektör) de 200'lük bir batch de aynı koddan geçiyor.
- **`if not 1 <= k <= n`**: Geçersiz k için hata veriyor. Numba kernel'leri sınır kontrolü yapmıyor; k = 0 bellek dışına okumaya yol açardı.
- **`Q @ self.X.T`**: Tüm skorlar tek bir BLAS çağrısıyla. Query'leri batch halinde vermek önemli: Phase 2C'de 20 query 362 GFLOP/s, 200 query 1.415 GFLOP/s vermişti.
- **`topk(...)`**: 8 veya daha fazla query varsa paralel (`prange`), daha azsa tek thread'li kernel. Tek query'de thread başlatmanın maliyeti kazançtan büyük.
- **Parçalı arama**: corpus `chunk_docs`'luk dilimler halinde işleniyor ve her dilimden top-k alınıyor. Bellekte hiçbir zaman tüm skor matrisi tutulmuyor.

---

## 2. Doğruluk kontrolü

| Kontrol | Sonuç |
|---|---|
| Tam mod, 20K, **200 query**, Phase 1 naive referansına karşı | En büyük skor farkı **8,4 × 10⁻⁷**. Top-10 listeleri **%100 birebir aynı** (sıra dahil) |
| Tam mod, 1K / 5K / 10K / 20K, naive arama şimdi yeniden çalıştırılarak (20 query) | En büyük fark 7,3–8,0 × 10⁻⁷. Tüm boyutlarda listeler %100 aynı |
| Hızlı mod (k = 300), 20K | recall@10 = **0,954**. Tasarım gereği yaklaşık; 10⁻⁵ toleransı aranmıyor |
| Parçalı ve parçasız arama, 200K | Skorlar ve doküman indeksleri **birebir aynı** |

Kalan ~10⁻⁷'lik fark float32 aritmetiğinden geliyor: baseline Python'un float64 sayılarıyla, final sistem float32 ile hesaplıyor. Skorlar 10⁻⁵ toleransının 12 kat altında ve hiçbir sıralamayı değiştirmiyor.

Bu kontrolün düzeni Phase 2B'deki kararla örtüşüyor: **tam mod** baseline ile birebir eşleşen moddur. **Hızlı mod** ise bir hız-doğruluk takası; onun kalitesini 10⁻⁵ ile değil recall ile ölçüyoruz.

---

## 3. Ölçeklenme

### 3.1 Ölçüm düzeni
- **Gerçek corpus:** 1K, 5K, 10K, 20K doküman (karıştırılmış corpus'un ilk N satırı).
- **Sentetik uzantı:** 50K, 100K, 200K. Gerçek vektörlerden rastgele biri seçiliyor, %5 gürültü ekleniyor ve tekrar normalize ediliyor. **Sadece zaman ölçümü için**; doğruluk ve kalite sonuçları bu verilerden çıkarılmadı. Amaç "10 kat daha fazla veri olsaydı ne olurdu?" sorusunu tahminle değil ölçümle cevaplamak.
- İki kullanım şekli: **200 query'lik batch**, ve **tek query** (20 query'yi tek tek arayıp ortalaması).
- Her ölçüm 2 warmup + 5 ölçüm, median.

### 3.2 Sonuçlar (ms)

| Doküman | Naive (200 query) | Tam, batch 200 | Hızlı, batch 200 | Tam, tek query | Hızlı, tek query | Corpus (tam) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 7.742 | 0,38 | 0,35 | 0,012 | 0,016 | 3 MB |
| 5.000 | 37.944 | 1,44 | 0,79 | 0,117 | 0,030 | 15 MB |
| 10.000 | 84.290 | 2,69 | 1,41 | **0,438** | 0,062 | 31 MB |
| 20.000 | 158.298 | **5,02** | **2,63** | 0,869 | 0,356 | 61 MB |
| 50.000 *(sentetik)* | — | 12,29 | 6,07 | 2,177 | 0,872 | 154 MB |
| 100.000 *(sentetik)* | — | 23,13 | 11,54 | 4,286 | 1,728 | 307 MB |
| 200.000 *(sentetik)* | — | 47,77 | 22,18 | 8,794 | 3,402 | 614 MB |

Naive sütunu Phase 1'den (20 query ölçülüp 200'e ölçeklendi).

Zamanın dağılımı (tam mod, 200 query):

| Doküman | Matris çarpımı | Top-k | Top-k payı |
|---:|---:|---:|---:|
| 1.000 | 0,22 ms | 0,10 ms | %31 |
| 20.000 | 4,69 ms | 0,28 ms | %6 |
| 200.000 | 46,74 ms | 2,27 ms | %5 |

![Figure 6](../figures/fig6_scaling.png)

> **Figure 6:** (a) Final sistem toplu aramada doğrusal ölçekleniyor: 5K'dan 200K dokümana kadar 1.000 doküman başına ~230–290 µs (200 query), ve gerçek boyutlarda naive baseline'dan 20.000–32.000 kat hızlı. Tek query de doğrusal ölçekleniyor; tek istisna, corpus'un 16 MB'lık L2 önbelleği aştığı 5K–10K arasındaki 3,7 katlık sıçrama. (b) Corpus büyüdükçe zamanın büyük kısmı matris çarpımına kayıyor (%69'dan %95'e); top-k artık bir bottleneck değil.

### 3.3 Sistem doğrusal ölçekleniyor mu?

**Toplu aramada evet.** 1.000 doküman başına süre 5K'da 287 µs, 20K'da 251 µs, 200K'da 239 µs; neredeyse sabit. Sebebi, 200 query'lik matris çarpımının **hesap gücüyle sınırlı** olması: her doküman sayısı bellekten bir kez okunup 200 query için kullanılıyor. 200K'da matris çarpımı 1,31 TFLOP/s'de çalışıyor; bu, Phase 2C'de ölçtüğümüz en yüksek değere (1,415 TFLOP/s) yakın. İş N ile doğrusal arttığı için süre de doğrusal artıyor.

1K'da doküman başına maliyet daha yüksek (379 µs). İş o kadar küçük ki top-k'nın thread başlatma maliyeti gibi sabit masraflar öne çıkıyor; top-k payının %31 olması da bunu gösteriyor.

**Tek query'de bir istisnayla evet.** Tek bir query için matris çarpımı bir matris-vektör çarpımına dönüşüyor. Corpus'taki her sayı bir kez okunup **tek bir** çarpmada kullanılıyor, dolayısıyla süreyi okuma hızı belirliyor:
- 5K'da corpus 15 MB ve L2 önbelleğe sığıyor: 15,4 MB / 0,117 ms ≈ **130 GB/s** etkin okuma hızı.
- 10K'da corpus 31 MB ve artık önbelleğe sığmıyor. Toplam süre 2 kat değil **3,7 kat** artıyor; doküman başına maliyet 23 ns'den 44 ns'ye çıkıyor (1,9 kat).
- 20K'dan 200K'ya kadar yine doğrusal: 61 MB / 0,87 ms ≈ 71 GB/s, 614 MB / 8,8 ms ≈ 70 GB/s. Ulaştığımız okuma hızı bu; ana belleğin teorik bant genişliğini ayrıca ölçmedik.
- Sıçramanın sebebinin L2 önbellek olduğu bir yorum: 5K ile 10K arasında ara ölçüm yapmadık, ama 15 MB'ın 16 MB'lık L2'ye sığıp 31 MB'ın sığmaması bu yorumla tutarlı.

Hızlı mod (300 boyut) aynı sıçramayı daha geç yaşıyor: 10K'da corpus 12 MB ve hâlâ önbellekte (0,062 ms); 20K'da 24 MB ve önbellekten taşıyor (0,356 ms, 5,7 kat). Bu, boyut indirgemenin asıl değerini gösteriyor: **tek query'de 2,4–2,6 kat** (20K'da 0,87'ye karşı 0,36 ms, 200K'da 8,8'e karşı 3,4 ms), çünkü bellekten okunacak veri 2,6 kat az.

### 3.4 Nerede bozuluyor?
1. **Önbellek sınırı (tek query):** Corpus L2'den taşınca tek query'nin doküman başına maliyeti ~2 katına, toplam süre 3,7 katına çıkıyor (Bölüm 3.3).
2. **Skor matrisi belleği (batch):** 200 × N × 4 byte. 20K'da 16 MB, 200K'da 160 MB, 2 milyonda 1,6 GB. Parçalı arama bunu çözüyor:

   | 200K doküman, 200 query | Süre | Aramanın tepe belleği |
   |---|---:|---:|
   | Parçasız | 49,9 ms | 160,6 MB |
   | Parçalı (20K'lık dilimler) | 52,1 ms | **16,9 MB** |

   Neredeyse aynı hız (%4 fark), aynı sonuç (skorlar ve indeksler birebir), 10 kat az bellek.
3. **RAM (corpus):** 200K'da 614 MB (hızlı modda 240 MB), 2 milyonda 6,1 GB. 16 GB RAM'li bir laptop'ta işletim sistemi ve modelle birlikte sınıra yaklaşılır.

### 3.5 10 kat daha fazla veriyle ne yapardık?
**10 kat (200K): mevcut sistem yetiyor.** Sentetik ölçüm: tam modda 200 query 48 ms, tek query 8,8 ms; hızlı modda 22 ms ve 3,4 ms. Etkileşimli kullanım için hâlâ çok hızlı. Yapacağımız tek değişiklik, belleği sınırlamak için parçalı aramayı açmak.

**100 kat (2 milyon) ve ötesi:**
- **Hızlı mod (SVD)** varsayılan olur: 6,1 GB yerine 2,4 GB, ve tek query'de bant genişliği kazancı.
- Corpus RAM'e sığmazsa diskteki memmap'ten **parça parça okuma** (Dask ya da kendi parça döngümüz). Kodumuz bunu zaten parça bazında yapıyor.
- Tek makine yetmezse corpus'u makinelere bölüp her birinde top-k almak ve sonuçları birleştirmek (Ray). Bizim `merge_topk` adımımız tam olarak bu birleştirme.
- Tam aramanın maliyeti N ile doğrusal. Milyonlarca dokümanda doğrusalın altında ölçeklenen **yaklaşık en yakın komşu** yöntemleri (HNSW, IVF) gerekir. Bedeli biraz recall kaybı; bu ödevde bir bonus maddesi olarak listeleniyor.

---

## 4. Genel karşılaştırma: tüm versiyonlar

Aynı görev: 200 query × 20.000 doküman, top-10. Hepsi aynı çalıştırmada, aynı koşullarda ölçüldü.

| Versiyon | Faz | Süre | Naive'e göre | Baseline'dan sapma |
|---|:-:|---:|---:|---:|
| Naive Python döngüleri | 1 | 154,5 s* | 1× | 0 |
| Numba `@njit` döngüleri + top-k | 2 | 1,366 s | 113× | 1,1 × 10⁻⁷ |
| Numba parallel + fastmath, paralel top-k | 2 | 35,8 ms | 4.321× | 1,4 × 10⁻⁷ |
| NumPy matris çarpımı + NumPy argpartition | 2 | 33,8 ms | 4.572× | 8,0 × 10⁻⁷ |
| **Final, tam** (matris çarpımı + Numba paralel top-k) | 4 | **4,84 ms** | **31.921×** | 8,4 × 10⁻⁷ |
| Final, hızlı (SVD k = 300) | 4 | 2,51 ms | 61.675× | yaklaşık (recall 0,95) |

\* 5 query ölçüldü ve 40 ile çarpıldı (süre query sayısıyla doğrusal; Phase 1). Phase 1'deki 20 query'lik ölçüm 158,3 s vermişti; o baseline'a göre tam mod 32.700 kat.

![Figure 7](../figures/fig7_overview.png)

> **Figure 7:** Aynı görevde performans hikayesinin her adımı: döngüleri derlemek 113 kat, SIMD ve 10 çekirdek 4.300 kat, BLAS 4.600 kat kazandırıyor. Ardından ortaya çıkan yeni bottleneck'i (top-k) kendi Numba kernel'imizle çözmek, baseline ile birebir aynı sonuçlarla ~32.000 kata çıkarıyor. SVD'li hızlı mod, recall@10 = 0,95 karşılığında ~62.000 kata ulaşıyor.

Hikaye Amdahl Yasası'nın iki kez işlediğini gösteriyor:
- Phase 1'de zamanın %99,5'i skorlamadaydı; sort %0,3'tü. Skorlamayı BLAS ile çözünce (33,8 ms) zamanın %87'si top-k'ya kaydı (Phase 2C).
- Top-k'yı Numba ile çözünce (4,84 ms) darboğaz tekrar matris çarpımına döndü: 20K'da zamanın %94'ü orada, ve matris çarpımı ölçtüğümüz en yüksek throughput'a yakın çalışıyor.

---

## 5. Canlı demo: `demo.py`

```bash
.venv/bin/python demo.py                      # interaktif, tam mod
.venv/bin/python demo.py --fast               # SVD k=300
.venv/bin/python demo.py --query "your text"  # tek query
```

Örnek çıktı (tek bir çalıştırma):

```
Ready: 20,000 arXiv abstracts (2025-26), mode = exact, model on cpu, loaded in 3.0 s

"efficient attention for long context language models"
  encode 19.5 ms | search 0.92 ms over 20,000 documents
  1. [0.782] Cost-Optimal Grouped-Query Attention for Long-Context Modeling
       arXiv:2503.09579  cs.CL  2025-03-12
  2. [0.756] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval
       arXiv:2501.15225  cs.CL  2025-01-25
  ...
```

Demo her query'de süreyi bir kez ölçüp yazdırıyor; bunlar benchmark değil, örnek. Benchmark'lı tek query arama süresi Bölüm 3.2'de: tam modda 0,87 ms, hızlı modda 0,36 ms.

Tasarım notları:
- **Her şey yerel:** Model HuggingFace'in yerel önbelleğinden, internete bağlanmadan yükleniyor (`HF_HUB_OFFLINE=1`). Sınıfta bağlantı sorunu demoyu etkilemez.
- **Warmup:** Açılışta iki deneme araması yapılıyor; model ve Numba kernel'leri ilk kullanıcı query'sinden önce hazır oluyor.
- **İki süre ayrı gösteriliyor:** cümleyi vektöre çevirme (embedding modeli) ve arama. FAQ'ya göre embedding süresi benchmark'a dahil değil, ama demoda dürüstçe ikisini de gösteriyoruz. Kullanıcının beklediği sürenin ~%95'i modelde.
- **Model CPU'da:** `src/extra_measurements.py` ile ölçtük (`results/extra_encode_latency.json`), her biri bir kez kodlanan 20 farklı query ile:

  | Cihaz | Medyan | En kısa | En uzun |
  |---|---:|---:|---:|
  | CPU | 17,3 ms | 15,8 ms | 19,2 ms |
  | Apple GPU (MPS) | 12,1 ms | 10,1 ms | **222,7 ms** |

  MPS medyanda daha hızlı, ama yeni bir girdi boyutu geldiğinde ara sıra derleme yapıp 200 ms'nin üstüne çıkıyor. Canlı demoda sabit ve öngörülebilir bir süre, birkaç ms'lik ortalama kazançtan daha önemli; bu yüzden CPU.

---

## 6. Yeniden üretmek için

```bash
.venv/bin/python src/phase4_final.py              # ~4 dk: doğruluk + ölçeklenme + genel karşılaştırma
.venv/bin/python src/phase4_final.py --plot-only  # sadece Figure 6 ve 7
.venv/bin/python src/extra_measurements.py encode # encode süresi, CPU ve MPS
```

---

## 7. Sunumda gelebilecek sorular

- **Final sistem baseline ile nasıl aynı sonucu veriyor?** Aynı cosine'i hesaplıyor: normalize edilmiş vektörlerin dot product'ı. Farklar float32 yuvarlamasından (≤ 8,4 × 10⁻⁷) ve hiçbir sıralamayı değiştirmiyor; 200 query'nin hepsinde top-10 listesi birebir aynı.
- **Neden iki mod var?** Tam mod baseline ile birebir eşleşiyor (ödevin doğruluk şartı). Hızlı mod %5 recall karşılığında toplu aramada ~1,9 kat, tek query'de ~2,4 kat hız ve 2,6 kat az bellek sunuyor.
- **Sistem doğrusal mı?** Toplu aramada evet (~250 µs / 1.000 doküman), çünkü matris çarpımı hesap gücüyle sınırlı. Tek query'de de, corpus'un L2 önbelleği aştığı nokta dışında.
- **5K'dan 10K'ya tek query neden 3,7 kat yavaşladı?** 5K'da corpus (15 MB) 16 MB'lık L2'ye sığıyor; 10K'da (31 MB) sığmıyor ve her query ana bellekten okunuyor. Doküman başına maliyet yaklaşık iki katına çıkıyor.
- **Batch neden tek query'den bu kadar verimli?** Bellekten okunan her sayı 200 query için kullanılıyor (hesap sınırlı). Tek query'de her sayı bir kez kullanılıyor (okuma hızı sınırlı). 200 query'yi tek tek aramak 200 × 0,87 ≈ 174 ms, toplu aramak 5 ms.
- **Sentetik veri neden kullanıldı?** Sadece 10 kat büyük bir corpus'ta zamanın nasıl davrandığını ölçmek için. Doğruluk ve kalite sonuçları yalnızca gerçek veriden.
- **Parçalı arama neden neredeyse aynı hızda?** Matris çarpımının toplam işi aynı; sadece skorlar parça parça üretilip hemen top-k'ya indiriliyor. Kazanç bellekte: 160 MB yerine 17 MB.
- **10 kat veri gelse ne yapardınız?** Mevcut sistem 200K'da 48 ms (batch), 8,8 ms (tek query) veriyor; parçalı aramayı açmak yeterli. 100 katta hızlı mod, diskten parça parça okuma, makineler arası bölme ve yaklaşık en yakın komşu (HNSW).
- **Demoda zamanın çoğu nereye gidiyor?** Cümleyi vektöre çevirmeye (~15–19 ms). Arama ~0,9 ms.
