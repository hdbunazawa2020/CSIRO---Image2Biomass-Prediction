# Progress.md

- 開発の進捗について 

| 日付 | タイトル | 内容 | 参照 |
| -- | -- | -- | -- |
| 25/12/20 | レポジトリ立ち上げ | レポジトリを立ち上げ. | - |
| 25/12/20 | EDA,前処理コード整備 | EDAとpreprocessを検討.| - |
| 25/12/27 | 前処理コード、学習コード作成 | convnextの学習コードを作成,2GPUで回せるようにした. | - |
| 25/12/28 | oof分析コード, Streamlit作成 | 推論した結果を後解析できるように整理. | - |

- 実験の進捗についてのメモ

| 日付      | 実験名    | データ                | CV        | モデル         | Ep  | 検討内容        | 結果                       | weighted_r2_score(CV) | LB |
| --  | --    | --    | --   | --    | --      | -- | --   | --   | --   |
| 25/12/27 | exp000   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 20  | まずはコード実装 | 幾らかのバグはあるが貫通はした. | -0.86 | - |
| 25/12/28 | exp001   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 20  | 評価指標のmetric見直し, コード理解とbug fix | Esが発動してEp13で終了 | -0.86 | - |
| 25/12/28 | exp002   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 20  | exp001+Es無効化 | 学習が飽和していない | −0.70 | - |
| 25/12/28 | exp003   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 500 | exp002+Ep:20->500 | Ep100超で学習が飽和 | 0.310 | - |
| 25/12/29 | exp004   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 100 | lr=1e-4で固定 | 学習は飽和 | 0.485 | - |
| 25/12/29 | exp005   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 100 | lr=5e-4で固定 | 学習は飽和せず | 0.268  | - |
| 25/12/29 | exp006   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 100 | exp004+混合loss（log+raw） | r2は上がらず、学習は途上 | 0.378  | - |
| 25/12/29 | exp007   | 000_preprocess_ver00 | 0/5fold   | convnext_base | 100 | exp006+vertical_flip(p=0.5) | exp007よりも学習遅い. augで多様性が出たかも？ | 0.374  | - |
| 25/12/29 | exp008   | 000_preprocess_ver00 | 0-4/5fold | convnext_base | 200 | wandb_sweepのBest条件トライ | - | -  | - |

- wandb_sweepについてのメモ

| 日付      | 実験タイミング | データ                 | CV        | モデル         | Ep  | 検討内容        | 結果                       | 
| --  | --    | --    | --    | --      | -- | -- | -- |
| 25/12/29 | exp007の後    | 000_preprocess_ver00 | 0/5fold   | convnext系 | 100  | lrは固定としつつ、色々な条件をsearch | r2は0.0~0.69まで色々 | 