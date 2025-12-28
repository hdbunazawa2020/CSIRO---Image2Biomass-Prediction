# Progress.md

- 開発の進捗について 

| 日付 | タイトル | 内容 | 参照 |
| -- | -- | -- | -- |
| 25/12/20 | レポジトリ立ち上げ | レポジトリを立ち上げ. | - |
| 25/12/20 | EDA,前処理コード整備 | EDAとpreprocessを検討.| - |
| 25/12/27 | 前処理コード、学習コード作成 | convnextの学習コードを作成,2GPUで回せるようにした. | - |
| 25/12/28 | oof分析コード, Streamlit作成 | 推論した結果を後解析できるように整理. | - |

- 実験の進捗についてのメモ

| 日付 | 実験名 | データ | モデル | 検討内容 | 結果 | weighted_r2_score(CV) | LB |
| --  | --    | --    | --    | --      | -- |
| 25/12/27 | exp000   | 000_preprocess_ver00 | convnext_base | まずはコード実装 | 幾らかのバグはあるが貫通はした. | -0.86 | - |
| 25/12/28 | exp001   | 000_preprocess_ver00 | convnext_base | 評価指標のmetric見直し, コード理解とbug fix | Esが発動してEp13で終了 | -0.86 | - |
| 25/12/28 | exp002   | 000_preprocess_ver00 | convnext_base | exp001+Es無効化 | 学習が飽和していない | −0.70 | - |
| 25/12/28 | exp003   | 000_preprocess_ver00 | convnext_base | exp002+Ep:20->200 | 0.31 | - |