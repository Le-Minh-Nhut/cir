# Bảng Xếp Hạng & Phân Rã Lỗi Trên EDIR (`70_10_20_seed42`)

- **Số model đã hoàn tất:** `2 / 15`

## 1. Tổng Quan (Full Corpus `28,645`, `1,003` test queries)

| Model | Best Ep | Overall `R@1` | `R@5` | `R@10` | `R@50` | `MRR` | Macro-15 `R@1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sprc** (`sprc_pretrain`) | 17 | **36.19%** | 85.84% | 96.51% | 99.20% | 0.5516 | 37.29% |
| **clip4cir** (`clip4cir_rn50x4`) | 5 | **27.02%** | 82.35% | 95.51% | 98.90% | 0.4785 | 27.39% |

## 2. Ma Trận `Recall@1` Theo 15 Subcategory

| Model | action | addition | cardinality | color | complex | material | remove | replace | shape | spatial | style | texture | time | viewpoint | weather |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sprc** | 45.8% | 61.4% | 36.8% | 48.3% | 27.5% | 37.7% | 17.7% | 54.8% | 23.0% | 32.8% | 40.6% | 27.9% | 40.0% | 25.0% | 40.0% |
| **clip4cir** | 37.3% | 35.1% | 26.3% | 41.4% | 24.4% | 23.0% | 19.4% | 29.0% | 34.4% | 26.2% | 25.0% | 32.8% | 30.0% | 11.7% | 15.0% |
