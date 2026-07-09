# Sunum Brifingi: Isaac Sim'de Çift Kollu SO-100 ile Deformable Kumaş Katlama Simülasyonu

> **Bu dosyanın amacı:** Bu belge, bir yapay zekâya (veya insana) verilip bu projeden bir **sunum (slayt destesi)** hazırlatmak için yazılmıştır. Projenin bağlamını, mimarisini, karşılaşılan problemleri ve çözümlerini kendi başına yeterli olacak şekilde anlatır — sunumu hazırlayanın kod tabanına erişimi olmadığı varsayılmıştır. En sonda önerilen slayt akışı vardır; birebir uyulması şart değildir.

---

## 1. Proje Özeti (Tek Paragraf)

İki adet **SO-100** robot kolunun (düşük maliyetli, 6 eklemli, açık kaynak hobby kolları), **NVIDIA Isaac Sim 5.1** içinde **deformable (partikül tabanlı) tişört ve benzeri giysileri katlamasını** sağlayan bir simülasyon ortamı geliştirildi. Katlama hareketleri **FoldNet** algoritmasının ürettiği tutma/bırakma (pick & place) hedefleriyle yönlendiriliyor; simülasyon ortamı bu hedefleri gerçek fizikle (PhysX PBD particle cloth) çalıştırıp **imitation learning için dataset** üretiyor. Projenin en zorlu ve öğretici kısmı, **robot gripper'ının kumaşı gerçekçi ve stabil şekilde "tutması"** problemi oldu: sahte (kinematik) tutmadan gerçek PhysX attachment'a uzanan bir mühendislik yolculuğu.

---

## 2. Bileşenler ve Mimari

### 2.1 Donanım modeli: SO-100
- 6 eklemli, STS3215 servo motorlu, paralel çeneli gripper'lı açık kaynak robot kol.
- Simülasyonda **iki kol** karşılıklı yerleştirilmiş; masa üzerinde giysi katlıyorlar.
- Gripper'ın kumaşa temas eden yüzeyi yaklaşık **1.5 cm** genişliğinde (bu ölçü, tutma modelinin boyutlandırılmasında doğrudan kullanıldı — bkz. §5.4).

### 2.2 Simülatör: NVIDIA Isaac Sim 5.1 (standalone Python)
- Fizik: **PhysX, PBD (Position-Based Dynamics) particle cloth** — kumaş, birbirine yay benzeri kısıtlarla bağlı binlerce partikülden oluşuyor.
- GPU pipeline: `eENABLE_DIRECT_GPU_API` aktif; rigid body durum okuma/yazma işlemleri USD üzerinden değil **tensor API** üzerinden yapılmak zorunda (bu, bir hata sınıfının kaynağı oldu — bkz. §6.5).

### 2.3 Katlama planlayıcı: FoldNet
- FoldNet, giysi mesh'i ve keypoint'ler (yaka, kol ucu, etek köşeleri: `l/r_sleeve_top`, `l/r_sleeve_bottom` vb.) üzerinden **katlama adımlarını** üretir: her adım "şuradan tut, şuraya taşı, bırak" şeklinde bir TCP (tool center point) yörüngesidir.
- Proje kapsamı gereği **FoldNet koduna ve mesh üretimine dokunulmadı**; tüm geliştirme Isaac tarafındaki runtime kodunda yapıldı (`isaac_sim/scripts/`).

### 2.4 Veri akışı
```
FoldNet giysi mesh'i (.obj + keypoint'ler)
        │
        ▼
garment_loader.py ──► PhysX PBD particle cloth (Isaac sahnesinde)
        │
        ▼
FoldNet politikası ──► pick/place TCP hedefleri
        │
        ▼
native_isaac_foldenv.py ──► kolları sür, kumaşı tut/taşı/bırak
        │
        ▼
generate_dataset_native.py ──► kamera görüntüleri + eklem verileri = dataset
```

### 2.5 Konfigürasyon felsefesi
- Bütün fizik ve tutma parametreleri **environment variable** olarak dışarı açıldı (40+ değişken: `ISAAC_GARMENT_STRETCH`, `ISAAC_BLOCK_ATTACHMENT_SIZE`, ...). Tek bir çalıştırma script'i (`run_sleeve_edge_grasp.sh`) güncel "en iyi bilinen" konfigürasyonu tutuyor. Bu, kod değişikliği olmadan hızlı deney iterasyonu sağladı.

---

## 3. Kronoloji / Fazlar (Git geçmişinden)

| Faz | İçerik |
|---|---|
| **Faz 0 — Robot kurulumu** | SO-100 USD modelleri; eklemlerin Physics Inspector'da geri fırlaması ve gripper'ın tabana gömülmesi düzeltildi; STS3215 motor modeli ve bilek (wrist) UVC kameraları eklendi; kol sarkması drive gain ayarıyla giderildi; siyah render veren bilek kamerası düzeltildi (gövde marker'ı lensi kapatıyordu). |
| **Faz 1 — Deformable kumaş** | FoldNet giysileri PhysX particle cloth olarak yüklendi; giysi fizik profilleri + CLI override'ları; düşerken buruşma (crumpling), kendi kendine kayma (self-sliding), sürtünme (PBD material binding ile) ve damping (0.35→0.50) düzeltmeleri. |
| **Faz 2 — Tutma (grasping) yolculuğu** | Projenin ana hikâyesi. Kinematik "sahte" tutmadan gerçek PhysX attachment'a geçiş. Detay: §4–§6. |

---

## 4. Ana Problem: Robot Gripper'ı Kumaşı Nasıl Tutar?

Rigid cisimlerde tutma kolaydır: sürtünme + kapanan çeneler yeter. **Particle cloth'ta ise gripper çeneleri partiküllerin arasından geçer** ya da kumaşı iter; sürtünmeyle güvenilir tutma pratik değildir. Simülasyon literatüründeki standart çözüm: tutma anında kumaş partiküllerini gripper'a **programatik olarak bağlamak**. Sorun, bunun *nasıl* yapıldığında.

### Deneme 1 — Kinematik ("sahte") tutma: her frame'de partikül pozisyonu dayatmak
Tutulan partiküllerin pozisyonları her simülasyon adımında elle TCP'ye taşındı. Sonuç: **fizik çözücüyle sürekli bilek güreşi.**
- PBD çözücü partikülleri kısıtlara göre bir yere koyuyor, biz zorla başka yere yazıyoruz → her frame'de yapay enerji pompalanıyor.
- Belirtiler: titreme (jitter), ani hız sıçramaları, kumaşın "şişmesi", bırakınca eski yerine lastik gibi geri fırlama ve **iki kez tam solver patlaması** (kumaş sahneden uçtu).
- Blend/max-step/catchup gibi ~10 yumuşatma parametresiyle semptomlar bastırılmaya çalışıldı; temel mimari sorun çözülmedi. **Ders: fizik motoruna karşı değil, fizik motorunun içinden çalışmak gerekir.**

### Deneme 2 — Semantik vertex seçimi (sleeve edge grasp)
Paralel bir iyileştirme: hangi partiküllerin tutulacağı. Tek partikül tutmak yerine, FoldNet keypoint'leri kullanılarak kolun **üst ve alt kenarları, ön ve arka katmanlarıyla birlikte** (≥4 anchor + komşuları) seçildi. TCP kol orta noktasına yaklaştığında otomatik devreye giren aktivasyon yarıçapı mantığı eklendi. Bu, "kolun önü/arkası sarkık kalıyor" problemini çözdü — ama tutmanın kendisi hâlâ sahteydi.

### Deneme 3 — Gerçek PhysX Attachment (nihai çözüm)
**GarmentLab** projesinin (particle cloth manipülasyonu için referans akademik kod tabanı) reçetesi incelenip uyarlandı:

1. Gripper TCP'sinde küçük bir **rigid cisim ("attachment block")** oluştur — görünmez olabilir, sadece fiziksel çapa.
2. Kumaş mesh'i ile blok arasında **`PhysxPhysicsAttachment`** tanımla, **`PhysxAutoAttachmentAPI`** ile PhysX'in bağlantı noktalarını kendisinin hesaplamasını sağla (`deformableVertexOverlapOffset` ile yakalama payı).
3. Blok **dinamik** (kinematik değil), kütlesi **1000 kg** (kumaş onu sürükleyemesin), **yerçekimi kapalı**.
4. Blok **pozisyon teleportuyla değil hız komutuyla** sürülür: `v = (hedef − mevcut) / (3·dt)` (P-kontrolcü). Böylece çözücü her şeyi kendi iç kısıtlarıyla çözer.
5. Blok çarpışması açık (vertex yakalama için gerekli) ama kumaş ve masayla **pair-filtered** (istenmeyen itme yok).

Sonuç: **enerji pompalama tamamen bitti.** Tutma, çözücünün *içinde* bir kısıt olduğu için taşıma sırasında kumaş doğal sarkıyor, bırakınca geri fırlamıyor, patlama yok. Kullanıcı testiyle doğrulandı.

---

## 5. İnce Ayar: Attachment Geometrisi (Son Faz)

Attachment çalıştıktan sonra kalan kalite sorunu: tutma **TCP'de değil** gibiydi ve **çok geniş bir hacmi** yakalıyordu. Kod analiziyle üç kök neden bulundu:

### 5.1 Kritik içgörü: PhysX auto attachment vertex seçimini tamamen yok sayar
Kodun özenle seçtiği 6 sleeve-edge vertex'i **sadece blok yerleşimi ve loglama** içindi. Gerçek kaynak (weld) kümesi = *blok çarpışma hacmi + overlap offset içine giren her kumaş vertex'i*. Yani tutma hassasiyeti tamamen **blok geometrisiyle** belirleniyor.

### 5.2 Yakalama hacmi çok genişti
Eski: 3 cm küp + 2 cm overlap → her yönde **~7 cm'lik yakalama kutusu**. Kolun yarısı kaynaklanıyordu.

### 5.3 Blok TCP'de değildi
Blok, seçilen vertexlerin **ortalama noktasına** (centroid) konuyordu — aktivasyon yarıçapı 6 cm olduğundan TCP'den 6 cm'ye kadar uzakta doğabiliyordu.

### 5.4 Çözüm (mevcut durum)
- **Küre** primitifi (küp yerine): köşesiz, her yönde eşit yakalama bölgesi.
- Çap **1.5 cm** = gripper'ın gerçek temas yüzeyi genişliği. Fiziksel gerçeklikten türetilmiş parametre — sunumda vurgulanmaya değer.
- Overlap 1 cm → efektif yakalama yarıçapı `0.0075 + 0.01 = 1.75 cm` (eski ~3.5 cm yarı-genişliğe karşı).
- Blok artık **tam TCP pozisyonunda doğuyor ve sıfır ofsetle TCP'yi izliyor**.
- Gözlemlenebilirlik: kırmızı debug küresi, oluşturma logunda `tcp_dist`, 8 frame sonra PhysX'in gerçekte kaç vertex kaynakladığını okuyan `attach diag: n=... extent=...` tanı satırı.

---

## 6. Sunumluk "Öğrenilen Dersler" (En Değerli Slaytlar)

1. **Fizik motoruna karşı değil, içinden çalış.** Dışarıdan pozisyon dayatmak enerji pompalar; çözücü-içi kısıt (attachment) + hız sürüşü stabil. Bu, projenin ana mühendislik dersi.
2. **API'nin gerçekte ne yaptığını doğrula.** Auto attachment'ın vertex seçimimizi yok sayması, semptomlardan değil kod/dokümantasyon analizinden bulundu.
3. **Fiziksel gerçeklikten parametre türet.** Blok çapı = gerçek gripper temas yüzeyi (1.5 cm). "Sihirli sayı" değil, ölçülebilir gerekçe.
4. **Gözlemlenebilirlik olmadan ayar yapılmaz.** Blok görünmezdi, kaynaklanan vertex sayısı bilinmiyordu → önce debug görselleri ve tanı logları eklendi, sonra ayar yapıldı.
5. **Referans implementasyon altın değerinde.** GarmentLab reçetesi (kütle 1000, yerçekimi kapalı, hız sürüşü, pair filtering) birçok deneme-yanılma turunu kısalttı. İlk PhysX denemesi başarısız olduğunda geri adım atmak yerine derinlemesine araştırma yapıldı ve reçete doğru uygulandı.
6. **GPU pipeline tuzakları:** `eENABLE_DIRECT_GPU_API` açıkken rigid body'ye USD üzerinden hız yazmak hata verir (`setLinearVelocity(): illegal`); tensor API view'ları kullanmak gerekir.
7. **Her şeyi env var'a çıkar.** 40+ parametre kod değişikliği olmadan denenebildi; "en iyi bilinen konfig" tek bir shell script'te versiyonlanıyor.

---

## 7. Güncel Durum ve Sonraki Adımlar

**Çalışıyor:** Çift kol sahnesi, deformable giysi yükleme, FoldNet güdümlü katlama adımları, gerçek PhysX attachment ile stabil tut-taşı-bırak (kullanıcı tarafından doğrulandı), dataset üretim pipeline'ı.

**Doğrulama aşamasında:** 1.5 cm TCP-merkezli küre geometrisi (tanı logları `tcp_dist ≈ 0` ve makul `n` değeri göstermeli).

**Bilinen risk / ayar kolları:** Yakalama bölgesi daraldığı için tutma anında kumaş TCP'den 1.75 cm'den uzaksa hiç vertex yakalanmaz (`n=0`) → `OVERLAP` 0.01→0.015 ilk müdahale. Tutma zayıfsa `SIZE` 0.015→0.02.

**Sonraki adımlar:** Geometri doğrulaması → tam katlama sekansının uçtan uca koşulması → çok kategorili dataset üretimi (tişört, pantolon, yelek, gömlek, kapüşonlu — 9 kategori × varyantlar) → imitation learning eğitimi.

---

## 8. Önerilen Slayt Akışı (12–15 slayt)

1. **Kapak** — "Simülasyonda Robotik Kumaş Katlama: SO-100 + Isaac Sim + FoldNet"
2. **Motivasyon** — Kumaş manipülasyonu neden zor (sonsuz serbestlik derecesi, deformable fizik); dataset'in imitation learning'deki rolü
3. **Sistem genel bakış** — §2.4'teki veri akışı diyagramı
4. **Bileşenler** — SO-100, Isaac Sim/PhysX PBD, FoldNet (birer cümle)
5. **Faz 0–1 hızlı özet** — robot kurulumu + kumaş fizik ayarları (buruşma/kayma/sürtünme düzeltmeleri)
6. **Ana problem** — gripper particle cloth'u neden tutamaz (görsel: çeneler partiküllerin arasından geçer)
7. **Deneme 1: kinematik tutma** — nasıl çalışır + neden patladı (enerji pompalama; jitter/geri fırlama/solver patlaması)
8. **Deneme 2: semantik vertex seçimi** — keypoint tabanlı kol-kenarı tutuşu
9. **Çözüm: gerçek PhysX attachment** — GarmentLab reçetesinin 5 maddesi (§4, Deneme 3)
10. **Karşılaştırma slaytı** — sahte vs. gerçek tutma (tablo: enerji pompalama / stabilite / geri fırlama / parametre sayısı)
11. **İnce ayar hikâyesi** — "attachment vertex seçimini yok sayar" içgörüsü; 7 cm kutu → 1.75 cm küre; blok TCP'ye taşındı
12. **Öğrenilen dersler** — §6'dan seçmeler
13. **Güncel durum + demo** — (varsa ekran görüntüsü/video: kırmızı küre gripper ucunda, kol kaldırılırken kumaş doğal sarkıyor)
14. **Sonraki adımlar** — dataset → öğrenme
15. **Teşekkür / sorular**

### Sunumu hazırlayacak yapay zekâya notlar
- Hedef kitle belirtilmemişse **teknik ama PhysX bilmeyen** bir kitle varsay (robotik/ML meslektaşları); PBD ve attachment kavramlarını bir cümleyle tanımla.
- Hikâye kemeri **problem→başarısız çözüm→içgörü→çalışan çözüm** olarak kurgulanmalı; en güçlü malzeme Deneme 1'in başarısızlığı ile Deneme 3'ün zarafeti arasındaki kontrast.
- Sayıları koru: kütle 1000 kg, çap 1.5 cm, yakalama yarıçapı 1.75 cm, 6 kat daralma (7 cm→1.75 cm hacim yarı-genişliği), 40+ env parametresi, 2 solver patlaması, 9 giysi kategorisi.
- Görsel yer tutucuları bırak: sahne ekran görüntüsü, kırmızı küre yakın çekim, katlama sekansı kareleri (sunum sahibi ekleyecek).

---

## 9. Mini Sözlük

| Terim | Anlamı |
|---|---|
| **PBD (Position-Based Dynamics)** | Kumaşı, pozisyon kısıtlarıyla birbirine bağlı partiküller olarak simüle eden yöntem |
| **TCP (Tool Center Point)** | Gripper'ın işlevsel uç noktası; tutma/bırakma hedefleri bu noktaya tanımlanır |
| **PhysX Attachment** | İki aktör (kumaş ↔ rigid blok) arasında çözücü-içi kalıcı bağ (kaynak/weld) |
| **Auto attachment** | PhysX'in bağlantı noktalarını geometrik örtüşmeden otomatik hesaplaması |
| **Overlap offset** | Blok yüzeyinden dışa doğru ek yakalama payı |
| **Pair filtering** | Belirli aktör çiftleri arasında çarpışmayı kapatma (bağ kalır, itme kalkar) |
| **Keypoint** | Giysi üzerindeki semantik nokta (kol ucu üst/alt kenarı, yaka vb.) |
| **Imitation learning** | Uzman gösterimlerinden (burada: simülasyon yörüngeleri) politika öğrenme |
| **GarmentLab** | Isaac Sim'de giysi manipülasyonu için referans alınan akademik kod tabanı |
