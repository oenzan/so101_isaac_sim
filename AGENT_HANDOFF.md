# SO100 Cloth-Folding — Ajan Devir Dosyası (Handoff)

> Bu dosya, `/home/ozan/Downloads/so100_ws` projesinde çalışacak yeni bir AI ajana
> bağlam aktarmak için yazıldı. Tarih: 2026-07-08, branch: `feature_cloth`.

## 1. Proje nedir?

Isaac Sim 5.1 standalone (`/home/ozan/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh`)
içinde çift SO-100 robot kolu, FoldNet garment'larını (tişört) katlıyor. Kumaş,
PhysX **PBD particle cloth** olarak simüle ediliyor. Amaç: FoldNet policy'sinin
ürettiği pick&place aksiyonlarıyla gerçekçi katlama + dataset üretimi
(`isaac_sim/scripts/generate_dataset_native.py`).

## 2. KIRMIZI ÇİZGİLER (mutlaka uy)

1. **Isaac Sim'i asla kendin başlatma.** Kullanıcı bütün sim koşularını kendisi
   çalıştırır; sen kodu hazırlarsın, o çalıştırıp log yapıştırır.
2. **V1 kapsamı = sadece Isaac tarafı** (`isaac_sim/scripts/`).
   `FoldNet_code/`, mesh üretimi ve `foldnet_garments/` içeriği **değiştirilmez**.
3. **Önce logla, sonra aksiyon al.** Kullanıcının yerleşik metodolojisi:
   "Önce loglayalım. Sebebini iyice tespit edip öyle aksiyon alalım." Hipotezle
   parametre kurcalama; enstrümante et, logla teşhis et, tek değişiklik yap,
   A/B için env-gate koy.
4. Kullanıcı Türkçe konuşur; teknik terimler İngilizce kalabilir.

## 3. Dosya haritası

| Dosya | Rolü |
|---|---|
| `isaac_sim/scripts/native_isaac_foldenv.py` | Ana env. Picker/grasp/release durum makinesi, block attachment sürüşü, ClothDiag + FoldDiag enstrümantasyonu. Neredeyse tüm iş burada. |
| `isaac_sim/scripts/run_sleeve_edge_grasp.sh` | Kanonik koşu script'i. **Bütün tuning env var'ları burada** — parametre değişikliği = bu dosyada export değiştirmek. |
| `isaac_sim/scripts/generate_dataset_native.py` | Entry point (script bunu çağırır). |
| `isaac_sim/scripts/garment_loader.py` | Mesh → PBD cloth yükleme, PBD materyal (friction burada bind edilir). |
| `isaac_sim/scripts/config.py` | Varsayılanlar (TABLE_HEIGHT, garment profilleri). Env override'lar bunları ezer. |
| `foldnet_garments/tshirt_sp_0/` | Test tişörtü: mesh.obj (5264 vert, 0.35 scale → 33.8×18.3 cm) + mesh_info.json (keypoint'ler: `l/r_sleeve_top`, `l/r_sleeve_bottom`, her biri [ön,arka] katman index'i). |

Test: `python -m py_compile isaac_sim/scripts/native_isaac_foldenv.py` her
değişiklikten sonra çalıştırılır (sim'i kullanıcı çalıştırdığı için tek hızlı doğrulama bu).

## 4. Kanıtlanmış fizik gerçekleri (yeniden tartışma)

- **`ISAAC_GARMENT_DAMPING=0.3` stabil; ≥0.8 patlar.** PBD spring damping,
  18 mg'lık particle'larda (0.09 kg / 5050 vert) enerji POMPALAR (k=c·dt/m
  overcorrection). İnstabilite önce dikişlerde (0.75–2 mm kısa kenarlar) çıkar.
  Damping ≤ ~0.5 sert tavandır. Teşhis aracı: ClothDiag.
- **Tişört kütlesi 90 g bilinçli olarak 6× ağır** (gerçekçi jersey ~15 g olurdu).
  Gerçekçi kütle particle'ı 3 mg yapar → stabilite 6× kötüleşir. 90 g'a dokunma;
  davranış oranlarla (mass ↔ stretch/bend/friction) birlikte tune edildi.
- **`solid_rest_offset` (0.0025) ≠ `contact_offset` (0.006)**: eşit olsalardı
  katmanlar 12 mm ayrık dururdu ve tişört kendiliğinden şişip kayardı.
- **Sürtünme (FRICTION=1.0) yeterli** — kumaş masaya düz temas ettiğinde milim
  kımıldamıyor. Kumaş kendiliğinden hareket ediyorsa sebep sürtünme değildir.

## 5. Teşhis araçları (kodda hazır)

### ClothDiag (`ISAAC_CLOTH_DIAG=1`, EVERY=30, SPIKE_VEL=0.25)
Periyodik `[ClothDiag]` raporu: ortalama/maks particle hızı, drift, top movers
(vertex index'leri — ardışık index kümesi = dikiş = damping instability parmak izi),
spike alarmı. Ham post-solver hızları okur (`_damp_cloth_velocities`'ten önce).

### FoldDiag (`native_isaac_foldenv.py` içinde, ClothDiag ile birlikte akar)
Katlama kalıcılığını ölçer:
- Grasp anında tutulan vertex COM'u snapshot'lanır; release anında
  `[FoldDiag:{el}] release@f=...: ... |xy|=...m dz=...m` satırı basılır
  (**dz = bırakma yüksekliği** — masadan kaç cm yukarıda bırakıldı).
- Sonraki raporlarda `back=...%`: tip'in XY yer değiştirmesinin
  release→grasp doğrusuna projeksiyonu ÷ katlama uzunluğu.
  **0% = katlama yerinde kaldı, 100% = tamamen geri açıldı.**
- `coher` (0–1): patch hız vektörlerinin tutarlılığı. ~1 = tutarlı kayma
  (geri açılma), ~0 = jitter (instabilite). `dir` = ortalama hız yönü.

## 6. Çözülmüş sorunlar (tarihçe)

1. **Kumaş kendiliğinden titreyip büzüşüyordu** → damping 0.8→0.3 (bkz. §4).
2. **Katlanan kol geri açılıyordu** (bu haftanın işi): FoldNet policy gripper'ı
   z≈0.04'te açıyor → katlama ucu masanın **+4.6 cm üstünde havada** bırakılıyor
   (FoldDiag: dz=+0.0458). Havadaki kemer sürtünmesiz; yerçekimi + bend gerilimi
   geri çözüyor (back% 17→38→103, coher 0.64–0.85). Elenen hipotezler: jitter,
   katman itmesi, tek seferlik recoil.
   **Fix = press-then-release**: OPEN geldiğinde attachment kopmaz; block XY
   sabitlenip masaya (+HEIGHT) indirilir, FRAMES kadar bastırılır, sonra kopar.
   Env: `ISAAC_RELEASE_PRESS_FRAMES=15` (0=kapalı, A/B), `_HEIGHT=0.010`,
   `_TIMEOUT=60`. **DURUM: implement edildi, kullanıcı validasyon koşusu bekleniyor.**
   Beklenen log: `[ReleasePress:right] start: ...` → `pressed after Nf` →
   FoldDiag release'te dz≈+0.01 ve back% ~0'da sabit.

## 7. Mevcut grasp/release mimarisi (özet)

- Grasp modu: `block_attachment` + PhysX auto attachment. TCP'de 1.5 cm'lik
  dynamic küre (mass 1000, gravity yok) velocity-drive ile sürülür
  (`_move_attachment_block`: `velocity=(target−cur)/(dt·3)`); PhysX attachment
  kumaşı solver içinde çeker.
- **Sadece SAĞ elde fiziksel grasp var** (`ISAAC_BLOCK_ATTACHMENT_HANDS=right`).
  Sol elin "katlaması" no-op'tur (~2 cm pasif sürüklenme) — bilinçli config.
- Sleeve edge grasp (`ISAAC_SLEEVE_EDGE_GRASP=1`): TCP kol midpoint'ine
  ACTIVATION_RADIUS (0.06) içinde yaklaşırsa `sleeve_top/bottom` [ön,arka]
  anchor'ları + komşuları seçilir; değilse box seçimine düşer.
- Release: `_finalize_release` → release damping (12 frame, vel×0.15) →
  attachment destroy. Press-then-release bunun önüne durum makinesi ekler
  (`_start_release_press` / `_update_release_press`).

## 8. Açık işler / bilinen pürüzler

1. **Press-then-release validasyonu** — kullanıcı koşacak; back% serisine göre
   HEIGHT/FRAMES ayarlanabilir. Ana metrik FoldDiag back%.
2. **Fold-2 grasp fallback**: 2. katlamada TCP–kol midpoint mesafesi
   (0.113–0.118 m) aktivasyon yarıçapını (0.06) aşıyor → shape seçimine düşüp
   sadece 2 vertex tutuyor. Henüz ele alınmadı.
3. **Sol elde attachment yok** → sol kol katlaması gerçekte olmuyor.
4. **Git hijyeni**: working tree'de commit'lenmemiş kritik değişiklikler var
   (FoldDiag + press-then-release, `native_isaac_foldenv.py` +
   `run_sleeve_edge_grasp.sh`). Bir kez harici bir checkout ile kısmen
   kaybedildiler — değişiklik yapmadan önce `git status`/`git diff` ile mevcut
   durumu koru. Staged `__pycache__/*.pyc` dosyaları da temizlenip
   .gitignore'a eklenmeli (kullanıcı onayıyla).

## 9. Güncel efektif parametreler (run_sleeve_edge_grasp.sh)

```
STRETCH=8000  BEND=35  SHEAR=50  DAMPING=0.3  FRICTION=1.0  MASS=0.09
CONTACT_OFFSET=0.006  SOLID_REST_OFFSET=0.0025
CLOTH_VEL_DAMP=0.95  CLOTH_MAX_VEL=0.5  RELEASE_DAMP_FRAMES=12
RELEASE_PRESS_FRAMES=15  RELEASE_PRESS_HEIGHT=0.010  RELEASE_PRESS_TIMEOUT=60
BLOCK_ATTACHMENT_HANDS=right  USE_PHYSX=1  sphere SIZE=0.015 OVERLAP=0.01
SLEEVE_EDGE_GRASP=1  VERTS_PER_EDGE=3  ACTIVATION_RADIUS=0.06
```

Yeni bir ayar eklerken kalıp: env var (`ISAAC_...`) + `_parse_*_env` ile
`__init__`'te oku + `run_sleeve_edge_grasp.sh`'e nedenini açıklayan yorumla
export ekle + A/B için kapatılabilir yap (0/boş = eski davranış).
