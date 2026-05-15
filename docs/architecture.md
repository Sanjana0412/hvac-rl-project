# Architecture Overview

See README.md for the full architecture diagram and data flow description.

## Component Map

| Component | File(s) | Role |
|-----------|---------|------|
| Simulator | `env.py: HVACSimulator` | 15-min HVAC physics dynamics |
| Environment | `env.py: HVACEnv` | State discretisation + reward shaping |
| RL Agents | `env.py: QLearningAgent, SARSAAgent` | Tabular Q-table policies |
| Training CLI | `train.py` | Train / tune / evaluate / register |
| Preprocessing | `data/preprocess.py` | Sensor cleaning + feature engineering |
| REST API | `app.py` | FastAPI inference server |
| Drift Monitor | `monitoring/monitor.py` | PSI-based distribution shift detection |
| CI/CD | `.github/workflows/ci_cd.yml` | Lint → test → preprocess → docker → retrain |
| K8s | `k8s/manifests.yaml` | Deployment, Service, Ingress, HPA |
| GitOps | `k8s/argocd-app.yaml` | ArgoCD auto-sync from main |
| Pipeline versioning | `dvc.yaml` | Reproducible stage DAG |
| Experiment tracking | MLflow (`mlruns/`) | Runs, metrics, model registry |
