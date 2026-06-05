# Data Dictionary : Deepfake and Real Images

> **Sumber dataset:** [Kaggle — Deepfake and Real Images (Manjil Karki)](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) + 4 dataset tambahan pada referesni  
> **Dataset asli:** OpenForensics (ICCV 2021) — Le et al.  
> **Output Data Wrangling:** `clean_data_sorted.csv` + `deepfake_clean_final.zip`

---

## Informasi Umum

| Atribut                   | Detail                                                                                      |
|---------------------------|---------------------------------------------------------------------------------------------|
| Total gambar awal         | 388.389 gambar (190.335 dataset awal + 198.054 dataset tambahan)                           |
| Total gambar setelah cleaning | 371.076 gambar                                                                          |
| Format gambar             | JPG (`.jpg`)                                                                                |
| Resolusi asli             | Bervariasi (width: 183–8.495px, height: 135–7.091px)                                       |
| Resolusi setelah packaging | 256 × 256 piksel (JPEG quality 85)                                                        |
| Mode warna                | RGB (3 channel)                                                                             |
| Kelas                     | Real, Fake                                                                                  |
| Class balance             | Seimbang (~50% Fake : ~50% Real)                                                            |
| Metode pemalsuan          | StyleGAN, Adversarial Latent Autoencoder (ALAE), GAN, Diffusion Model                      |
| Sumber gambar real        | Google Open Images + berbagai sumber dari 4 dataset tambahan                               |
| Output CSV                | `clean_data_sorted.csv`                                                                     |
| Output ZIP                | `deepfake_clean_final.zip` (4,35 GB)                                                       |

---

## Struktur Folder Dataset Awal (Manjil Karki)

Label pada dataset ini ditentukan secara implisit dari nama folder pada dataset. Struktur folder dipertahankan apa adanya tanpa dilakukan pembagian ulang.

```
Dataset/
├── Train/                  # 140.002 gambar
│   ├── Real/               # 70.001 gambar wajah asli
│   └── Fake/               # 70.001 gambar deepfake
├── Validation/             # 39.428 gambar
│   ├── Real/               # 19.787 gambar wajah asli
│   └── Fake/               # 19.641 gambar deepfake
└── Test/                   # 10.905 gambar
    ├── Real/               # 5.413 gambar wajah asli
    └── Fake/               # 5.492 gambar deepfake
```

---

## Struktur Folder Dataset Tambahan (4 Sumber)

Label pada keempat dataset tambahan dideteksi otomatis berdasarkan nama folder induk menggunakan keyword matching. Dataset tambahan dibagi ulang dengan proporsi Train 73,5% / Validation 20,7% / Test 5,8%.

```
chuneeb/deepfake-detection-dataset-2026/
├── fake/                   # gambar deepfake
└── real/                   # gambar wajah asli

saurabhbagchi/deepfake-image-detection/
├── Fake/                   # gambar deepfake
└── Real/                   # gambar wajah asli

xhlulu/140k-real-and-fake-faces/real_vs_fake/
├── fake/                   # gambar deepfake
└── real/                   # gambar wajah asli

prithivsakthiur/deepfake-vs-real-60k/
├── Fake/                   # gambar deepfake
└── Real/                   # gambar wajah asli
```

---

## Distribusi Data per Split

### Dataset Awal (Manjil Karki)

| Split      | Total       | Real    | Fake    | Proporsi      |
|------------|-------------|---------|---------|---------------|
| Train      | 140.002     | 70.001  | 70.001  | 50% / 50%     |
| Validation | 39.428      | 19.787  | 19.641  | 51% / 49%     |
| Test       | 10.905      | 5.413   | 5.492   | 49% / 51%     |
| **Total**  | **190.335** | **95.201** | **95.134** | **51% / 49%** |

### Dataset Tambahan (4 Sumber, setelah splitting)

| Split      | Total       | Proporsi  |
|------------|-------------|-----------|
| Train      | 145.569     | 73,5%     |
| Validation | 40.997      | 20,7%     |
| Test       | 11.488      | 5,8%      |
| **Total**  | **198.054** | **100%**  |

### Dataset Gabungan (setelah cleaning & deduplikasi)

| Split      | Total       | Real    | Fake    | Proporsi       |
|------------|-------------|---------|---------|----------------|
| Train      | 268.659     | 125.749 | 142.910 | 47% / 53%      |
| Validation | 80.024      | 39.989  | 40.035  | 50% / 50%      |
| Test       | 22.393      | 11.244  | 11.149  | 50% / 50%      |
| **Total**  | **371.076** | **176.982** | **194.094** | **48% / 52%** |

---

## Data Dictionary : `clean_data_sorted.csv`

File CSV ini adalah output utama dari proses data wrangling. Setiap baris merepresentasikan satu gambar yang telah lolos validasi dari seluruh 5 dataset sumber.

| Kolom      | Tipe        | Contoh Nilai                                                        | Deskripsi |
|------------|-------------|---------------------------------------------------------------------|-----------|
| `split`    | Categorical | `Train`, `Validation`, `Test`                                       | Split dataset. Untuk dataset awal mengikuti struktur folder original. Untuk dataset tambahan ditentukan oleh hasil `split_and_record()` dengan rasio 73,5/20,7/5,8. |
| `label`    | Categorical | `Real`, `Fake`                                                      | Label kelas dinormalisasi menggunakan `.strip().capitalize()` untuk konsistensi. `Real` = wajah asli. `Fake` = wajah sintetis. |
| `filename` | String      | `img_0001.jpg`                                                      | Nama file gambar. |
| `filepath` | String      | `/kaggle/input/datasets/.../Train/Real/img_0001.jpg`                | Path absolut menuju file gambar di lingkungan Kaggle. Mencakup path dari 5 dataset sumber yang berbeda. |

---

## Data Dictionary : Properti Gambar

| Atribut              | Tipe    | Nilai                              | Deskripsi |
|----------------------|---------|------------------------------------|-----------|
| `image_width`        | Integer | `256` (setelah packaging)          | Lebar gambar dalam piksel setelah proses standarisasi dengan `Image.thumbnail((256, 256))` pada tahap packaging. |
| `image_height`       | Integer | `256` (setelah packaging)          | Tinggi gambar dalam piksel setelah proses standarisasi. |
| `color_mode`         | String  | `RGB`                              | Mode warna 3 channel. Dikonversi eksplisit dengan `.convert('RGB')` saat assessing dan packaging. |
| `pixel_range`        | Integer | `0 – 255`                          | Rentang nilai piksel raw sebelum preprocessing model. Konversi ke float dilakukan di tahap feature engineering. |
| `mean_rgb`           | Float   | `[0.4806, 0.4118, 0.3799]`         | Rata-rata nilai piksel per channel (R, G, B), dihitung dari seluruh 388.389 gambar dataset gabungan. |
| `std_rgb`            | Float   | `[0.2801, 0.2598, 0.2561]`         | Standar deviasi nilai piksel per channel (R, G, B), dihitung dari seluruh dataset gabungan. |

---

## Preprocessing & Transformasi

| Langkah | Fungsi            | Parameter                          | Deskripsi |
|---------|-------------------|------------------------------------|-----------|
| 1       | `Image.thumbnail` | `(256, 256), Image.LANCZOS`        | Resize gambar ke maksimal 256×256 piksel dengan mempertahankan aspek ratio. Dilakukan pada tahap packaging sebelum dikemas ke ZIP. |
| 2       | `img.save`        | `format=JPEG, quality=85, optimize=True` | Kompres gambar ke format JPEG dengan quality 85 untuk efisiensi storage. Menghasilkan rata-rata ~14KB per gambar dari sebelumnya ~50–90KB. |

---

## Tahapan Data Wrangling

| Tahap          | Aktivitas                                                    | Temuan / Hasil |
|----------------|--------------------------------------------------------------|----------------|
| **Gathering**  | Load path dataset awal dari Kaggle                           | Dataset awal terstruktur dalam 3 split: Train (140.002), Validation (39.428), Test (10.905) |
|                | Gather 4 dataset tambahan via `gather_extra_images()`        | 198.054 gambar berlabel terkumpul (Fake: 99.143, Real: 98.911) |
|                | Split dataset tambahan (73,5% / 20,7% / 5,8%)               | Train: 145.569 / Validation: 40.997 / Test: 11.488 dengan `random.seed(42)` |
|                | Gabungkan ke `all_paths`                                     | Total 388.389 path gambar |
| **Assessing**  | Cek corrupt files & blank images (PIL, variance < 5)         | Corrupt: 0 / Blank: 0 — seluruh dataset valid |
|                | Cek duplikasi (MD5 hash, chunk 65.536 bytes)                 | 17.322 pasangan duplikat ditemukan |
|                | Cek resolusi & aspect ratio                                  | Width: 183–8.495px / Height: 135–7.091px / Aspect ratio: 0,3278–2,9979 |
|                | Distribusi kelas (dengan normalisasi capitalize)             | Fake: 194.277 / Real: 194.112 (~50/50) |
| **Cleaning**   | Hapus corrupt & blank images                                 | 0 gambar dihapus — seluruh 388.389 valid |
|                | Deduplikasi berdasarkan MD5 hash                             | 17.313 gambar dihapus → 371.076 gambar unik |
|                | Normalisasi label `.strip().capitalize()`                    | Konsistensi 2 kelas: `Fake` dan `Real` |
| **Packaging**  | Resize ke 256×256 + kompres JPEG Q85                         | Ukuran rata-rata turun dari ~50–90KB → ~14KB per gambar |
|                | Kemas ke `deepfake_clean_final.zip`                          | 371.076 gambar / 4,35 GB |

---

## Output Files

| File                        | Format | Ukuran    | Deskripsi |
|-----------------------------|--------|-----------|-----------|
| `clean_data_sorted.csv`     | CSV    | ~46,67 MB | Metadata seluruh 371.076 gambar valid: kolom `split`, `label`, `filename`, `filepath`. Terurut berdasarkan split → label → filename. |
| `deepfake_clean_final.zip`  | ZIP    | ~4,35 GB  | Arsip seluruh gambar valid dengan struktur internal `split/label/filename.jpg`. Resolusi 256×256, JPEG quality 85. Siap digunakan oleh tim AI Engineer. |

---

## Referensi

- Manjil Karki. *Deepfake and Real Images*. Kaggle, 2022. https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- Le, T.-N., Nguyen, H. H., Yamagishi, J., & Echizen, I. *OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild*. ICCV, 2021. https://arxiv.org/abs/2107.14480
- OpenForensics dataset (v1.0.0). Zenodo. https://zenodo.org/records/5528418
- Chuneeb. *Deepfake Detection Dataset 2026*. Kaggle, 2026. https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026
- Bagchi, S. *Deepfake Image Detection*. Kaggle. https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection
- Xhlulu. *140k Real and Fake Faces*. Kaggle. https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- Prithivsakthiur. *Deepfake vs Real 60k*. Kaggle. https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k
- Remonggkircop. *Data Wrangling Capstone Nambah Data* (Notebook). Kaggle, 2026. https://www.kaggle.com/code/remonggkircop/data-wrangling-capstone-nambah-data
- Remonggkircop. *Deepfake Clean Final* (Dataset). Kaggle, 2026. https://www.kaggle.com/datasets/remonggkircop/deepfake-clean-final