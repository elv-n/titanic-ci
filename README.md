# Workflow CI - Titanic (SMSML)

Continuous Integration untuk melatih ulang model Titanic memakai **MLflow
Project**, menyimpan artefak, lalu membangun **Docker Image** dan push ke
**Docker Hub** melalui `mlflow models build-docker`.

## Struktur

```
Workflow_CI/
├── .github/workflows/ci.yml        # workflow CI (GitHub Actions)
├── MLProject/
│   ├── MLProject                   # definisi MLflow Project + entry point
│   ├── conda.yaml                  # environment training
│   ├── modelling.py                # skrip training (entry point `main`)
│   ├── dataset_preprocessing/      # dataset siap latih (X/y train & test)
│   └── DockerHub.txt               # tautan & cara pakai Docker Image
└── README.md
```

## Menjalankan secara lokal

```bash
cd MLProject
mlflow run . --env-manager=local
```

Output: model + metrik tersimpan di `MLProject/mlruns/`, dan `run_id` ditulis ke
`MLProject/run_id.txt`. Parameter dapat dikustom, contoh:

```bash
mlflow run . --env-manager=local -P n_estimators=200 -P max_depth=5
```

## Build Docker Image dari model

```bash
cd MLProject
mlflow models build-docker \
  --model-uri "runs:/$(cat run_id.txt)/model" \
  --name "elivkurniawan/titanic-ci:latest"
```

## Konfigurasi CI (GitHub Actions)

Workflow `ci.yml` berjalan otomatis pada `push`/`pull_request` ke `main`/`master`
maupun manual (`workflow_dispatch`). Tahapannya:

1. Checkout repo + setup Python 3.12.
2. Install MLflow & dependency.
3. `mlflow run .` (training + logging artefak ke `mlruns/`).
4. Upload artefak (`mlruns/`, `run_id.txt`) sebagai artefak workflow.
5. Login Docker Hub.
6. `mlflow models build-docker` lalu `docker push`.

### Secrets yang diperlukan

Atur di **Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
| --- | --- |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN`    | Access Token Docker Hub (Account Settings → Security) |

Jika kedua secret belum diisi, step login & build-docker otomatis di-skip
sehingga workflow tetap hijau (training + upload artefak tetap berjalan).
