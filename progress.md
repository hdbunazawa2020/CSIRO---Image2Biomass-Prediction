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
| 25/12/29 | exp008   | 000_preprocess_ver00 | 0-4/5fold | convnext_base | 200 | wandb_sweepのBest条件トライ | 好スコア再現. Foldで結構揺れる. Foldにより得意な種が変わる. 学習は飽和していない. | 0.681(0.557~0.732)  | 0.55 |
| 25/12/29 | exp009   | 000_preprocess_ver00 | 0-4/5fold | convnext_base | 200 | exp008に対し、bs32->16(sweepに合わせる) | Foldで結構揺れる. bs=32の方が学習安定性はありそう. | 0.688(0.688~0.777)  | 0.56 |
| 25/12/30 | exp010   | 000_preprocess_ver00 | 0-4/5fold | convnext_base | 200 | exp008に対し、img_sizeを1x2にする | Foldごとのばらつきが提言された. | 0.723(0.687-0.792)  | 0.54 |
| 25/12/30 | exp011   | 000_preprocess_ver00 | 0-4/5fold | convnext_small | 200 | sweepのbest | LB bestになった. まだCV/LBの乖離がある. | 0.738(0.657-0.797)  | 0.58 |
| 25/12/30 | exp012   | 000_preprocess_ver00 | 0-4/5fold | convnext_small | 200 | BiomassConvNeXtMILHurdle | CV, LBともにあまり上がらず. 0gの予測がズレる問題あり. | 0.657(0.533-0.747)  | 0.51 |
| 25/12/31 | exp013   | 000_preprocess_ver00 | 0-4/5fold | convnext_small | 200 | exp12+lambda_presence(0.2->0.5) | スコアは悪化 |0.648 | 0.47  | 
| 25/12/31 | exp014   | 000_preprocess_ver00 | 0-4/5fold | convnext_small | 200 | exp12+lambda_presence(0.2->1.0) | スコアはさらに悪化 | 0.643 | 0.50  | 
| 26/01/01 | exp015   | 000_preprocess_ver00 | 0-4/5fold | convnext_small | 200 | exp12+sweep_best | - | 0.719  | 0.50 |
| 26/01/01 | exp016   | 001_preprocess_ver01 | 0-3/4fold | convnext_small | 200 | exp11+dataset変更 | - | 0.737  | 0.62 |
| 26/01/01 | exp017   | 001_preprocess_ver01 | 0-3/4fold | convnext_small | 200 | exp15+dataset変更 | - | 0.699  | 0.51 |



- wandb_sweepについてのメモ

| 日付      | 実験タイミング | データ                 | CV        | モデル         | Ep  | 検討内容        | 結果                       | 
| --  | --    | --    | --    | --      | -- | -- | -- |
| 25/12/29 | exp007の後    | 000_preprocess_ver00 | 0/5fold   | convnext系 | 100  | lrは固定としつつ、色々な条件をsearch | r2は0.0~0.69まで色々 | 
| 25/12/30 | exp010の後    | 000_preprocess_ver00 | 0/5fold   | convnext系 | 100  | 色々な条件をsearch | - | 
| 25/12/31 | exp012の後    | 000_preprocess_ver00 | 0/5fold   | convnext_small | 100  | BiomassConvNeXtMILHurdleの条件をsearch | - | 