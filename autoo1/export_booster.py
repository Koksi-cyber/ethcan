import os, glob, xgboost as xgb

os.makedirs("exported_models", exist_ok=True)
ok=False
for raw in sorted(glob.glob("extracted_boosters/booster_*.raw")):
    print(f"[INFO] trying {raw}")
    buf = open(raw,"rb").read()
    bst = xgb.Booster()
    try:
        bst.load_model(bytearray(buf))   # load from memory buffer
        print("[PASS] loaded from bytes")
        base=os.path.splitext(os.path.basename(raw))[0]
        bst.save_model(f"exported_models/{base}.ubj")      # binary JSON
        bst.save_model(f"exported_models/{base}.json")     # plain JSON
        open(f"exported_models/{base}_config.json","w").write(bst.save_config())
        print(f"[SAVED] exported_models/{base}.ubj/.json and _config.json")
        ok=True
        break
    except Exception as e:
        print("[FAIL] load/export:", repr(e))
if not ok:
    raise SystemExit(1)
