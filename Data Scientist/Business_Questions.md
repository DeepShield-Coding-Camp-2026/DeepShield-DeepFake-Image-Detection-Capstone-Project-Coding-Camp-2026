# Business Questions : DeepShield - Analisis Dataset Deepfake 2020 - 2026

---

## 1. Distribusi & Keseimbangan Data

**Pertanyaan:**
Bagaimana proporsi kelas antara gambar **"Fake"** dan **"Real"** pada dataset Deepfake tahun 2020-2026, dan apakah diperlukan penanganan _imbalanced data_ (seperti SMOTE atau undersampling) sebelum melatih model?

**Rencana Analisis:**
- Bar chart distribusi kelas
- Perhitungan rasio antar kelas
- Persentase tiap kelas
- Selisih jumlah sampel

---

## 2. Karakteristik Piksel

**Pertanyaan:**
Bagaimana perbedaan distribusi tingkat pencahayaan (*brightness*) dan kontras (*contrast*) antara gambar **"Fake"** dan **"Real"** pada dataset tahun 2020 - 2026, dan apakah perbedaan tersebut mengharuskan penerapan teknik augmentasi/manipulasi data (seperti *color jittering*) selama pelatihan model?

**Rencana Analisis:**
- Histogram distribusi piksel per kelas
- Boxplot *brightness* per kelas

---

## 3. Distribusi Warna (RGB)

**Pertanyaan:**
Apakah terdapat anomali atau pergeseran distribusi nilai intensitas warna pada channel RGB (*Red, Green, Blue*) gambar **"Fake"** dibandingkan **"Real"** di dataset tahun 2020 - 2026, yang dapat dimanfaatkan sebagai penentuan arsitektur layer (seperti *Channel Attention*) pada model pendeteksi?

**Rencana Analisis:**
- KDE plot distribusi R, G, B per kelas secara terpisah
- Mean dan standar deviasi tiap channel per kelas
