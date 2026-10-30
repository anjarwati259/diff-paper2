Output `src/data/make_masks.py`:
`{dataset}/{MAR|MCAR}{ratio}/seed{k}/train_mask.npy`, `val_mask.npy`, `test_mask.npy`
+ `mekanisme.json` (untuk MAR: kolom selalu teramati, bobot, intercept, proporsi missing aktual).
True = hilang. Mask hanya mencakup kolom fitur (bukan y).
