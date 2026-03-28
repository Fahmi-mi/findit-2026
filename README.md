# FindIT 2026 - Face Anti-Spoofing Classification

## Ringkasan

Proyek ini membangun pipeline klasifikasi gambar wajah anti-spoofing dengan 6 kelas:

- `realperson`
- `fake_unknown`
- `fake_mask`
- `fake_screen`
- `fake_mannequin`
- `fake_printed`

Tujuan utama kompetisi adalah memaksimalkan **Macro F1** pada data test.

## Constraint Dan Setup

- Metric utama: Macro F1
- Seed global: `42`
- Hardware target: RTX 3050 6GB (VRAM-safe first)
- Batas eksperimen praktis: <= 3 jam per run
- Tidak pakai external data
- Logging/tracking eksperimen: file artifact + nama run konsisten

## Struktur Proyek Penting

- `notebooks/01_eda_data_audit.ipynb`: audit data awal
- `notebooks/02_data_split_prep.ipynb`: cleaning + split anti-leak
- `notebooks/00_main_pipeline_submission.ipynb`: training baseline, evaluasi OOF, submission, ensemble
- `data/processed/clean/`: metadata bersih hasil prep
- `data/splits/`: file split final
- `experiments/checkpoints/`: checkpoint model per run
- `experiments/oof_predictions/`: OOF prediction per run
- `output/submissions/`: submission per run

## Data Audit (EDA) - Hasil Utama

Audit dilakukan pada train dan test untuk memastikan kualitas data dan risiko leakage.

Temuan utama:

- Train images: `1652`
- Test images: `404`
- Distribusi kelas train tidak seimbang (terbesar `realperson`, terkecil `fake_printed`)
- Tidak ditemukan file gambar korup
- Ditemukan banyak duplikasi berbasis hash (md5)
- Ditemukan overlap hash antara train dan test

Implikasi:

- Split harus group-aware berdasarkan hash agar duplikasi tidak bocor antar fold
- Baris ambigu (hash sama, label beda) harus dibuang sebelum training

## Data Cleaning Dan Split Prep

Notebook `02_data_split_prep.ipynb` menjalankan langkah berikut:

1. Recompute hash md5 untuk train/test
2. Deteksi hash ambigu (konten sama, label berbeda)
3. Drop baris ambigu dari train
4. Dedup train by hash
5. Buat 5-fold `StratifiedGroupKFold` (group = hash)
6. Simpan artifact metadata bersih dan split final

Output penting yang dihasilkan:

- `data/processed/clean/train_metadata_with_hash.csv`
- `data/processed/clean/test_metadata_with_hash.csv`
- `data/processed/clean/train_ambiguous_hash_rows.csv`
- `data/processed/clean/train_dedup_clean.csv`
- `data/processed/clean/train_test_overlap_hash_rows.csv`
- `data/splits/train_5fold_stratified_group_seed42.csv`
- `data/splits/split_prep_summary.csv`

Validasi split:

- Hash leakage antar fold: `0`
- Fold tetap seimbang secara kelas (stratified)

## Arsitektur Pipeline Training

Main pipeline (`00_main_pipeline_submission.ipynb`) mencakup:

- Config terpusat (`CFG` dataclass)
- Seed reproducibility
- Dataset + augmentasi (`albumentations`)
- Backbone model:
  - `convnext_tiny`
  - `efficientnet_b3`
- Loss: weighted cross entropy
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- AMP mixed precision
- Early stopping
- Full CV 5-fold
- Simpan OOF, confusion matrix, submission, config run

### Catatan Stabilitas Windows Notebook

Untuk mencegah freeze dataloader (train 0% stuck), worker diset:

- `num_workers = 0`
- `num_workers_val = 0`
- `num_workers_test = 0`

Notebook juga menyediakan **Dataloader Sanity Check** sebelum training penuh.

## Definisi Tingkatan Output Model Saat Ini

Output eksperimen sekarang dibagi menjadi 3 level:

1. **Baseline model awal**
   Run pembuka untuk validasi pipeline end-to-end.

2. **Baseline model kuat (single-model terbaik)**
   Single model hasil tuning terbaik, jadi anchor utama non-ensemble.

3. **Output akhir (ensemble weighted combine)**
   Gabungan prediksi dari beberapa run/model untuk dorong skor final.

> Inti konsep: yang digabung pada ensemble adalah **prediksi output**, bukan merge parameter/weight internal model.

## Ensemble Experiments (B, C, D)

Di notebook utama, ensemble diuji dengan skenario:

- `ens1_DB_06_04` -> D:0.6 + B:0.4
- `ens1_DB_07_03` -> D:0.7 + B:0.3
- `ens1_DB_05_05` -> D:0.5 + B:0.5
- `ens2_DC_05_05` -> D:0.5 + C:0.5
- `ens3_DBC_05_03_02` -> D:0.5 + B:0.3 + C:0.2

Pemilihan run:

- `B`: convnext_tiny img288 (run terbaru)
- `C`: efficientnet_b3 img288 (run sebelumnya)
- `D`: efficientnet_b3 img288 (run terbaru/terkuat)

Hasil ranking OOF (macro F1) yang sudah dihitung:

- `ens3_DBC_05_03_02`: `0.921807` (terbaik)
- `ens1_DB_05_05`: `0.920166`
- `ens1_DB_07_03`: `0.917720`
- `ens1_DB_06_04`: `0.917268`
- `ens2_DC_05_05`: `0.915378`

Artifact ensemble tersimpan di:

- `experiments/oof_predictions/ensemble/`
- `output/submissions/ensemble/`

## Alur Eksekusi Yang Direkomendasikan

Urutan kerja notebook:

1. `01_eda_data_audit.ipynb`
2. `02_data_split_prep.ipynb`
3. `00_main_pipeline_submission.ipynb`

Prinsip submit harian:

- Simpan 1 single-model terbaik sebagai anchor
- Uji 1-2 ensemble terbaik di leaderboard
- Kunci kandidat dengan performa public LB paling stabil

## Dependensi

Dependensi utama dikelola via `pyproject.toml`:

- `torch`, `torchvision`
- `albumentations`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`, `jupyter`, `ipykernel`, `tqdm`

## Status Proyek (Per 2026-03-28)

- Pipeline EDA -> split -> train -> OOF -> submission: **selesai dan berjalan**
- Baseline minimum target sudah tercapai
- Ensemble kandidat final sudah dihasilkan
- Tahap berikutnya: finalisasi pilihan berdasarkan public leaderboard
