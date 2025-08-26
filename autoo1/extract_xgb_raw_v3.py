import sys, pickle
from pathlib import Path

PKL = Path("eth_signal_model_retrained.pkl")

class DummyBooster:
    def __init__(self,*a,**k): self.state=None
    def __setstate__(self,state): self.state = state  # capture raw bytes

class DummyEstimator:
    def __init__(self,*a,**k): self.state=None
    def __setstate__(self,state): self.state = state

class InterceptUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Intercept ONLY xgboost + sklearn. Leave numpy alone.
        if module.startswith("xgboost"):
            return DummyBooster if name == "Booster" else DummyEstimator
        if module.startswith("sklearn"):
            return DummyEstimator
        return super().find_class(module, name)

def walk(obj, out):
    if isinstance(obj, DummyBooster):
        out.append(obj); return
    if isinstance(obj, dict):
        for v in obj.values(): walk(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj: walk(v, out)
    else:
        for a in dir(obj):
            if a.startswith("__"): continue
            try: v = getattr(obj, a)
            except Exception: continue
            walk(v, out)

def to_bytes(state):
    if isinstance(state, (bytes, bytearray, memoryview)): return bytes(state)
    if isinstance(state, dict):
        for v in state.values():
            if isinstance(v, (bytes, bytearray, memoryview)): return bytes(v)
    if isinstance(state, (list, tuple)):
        for v in state:
            if isinstance(v, (bytes, bytearray, memoryview)): return bytes(v)
    return None

def main():
    if not PKL.exists():
        print("[FAIL] missing", PKL, file=sys.stderr); sys.exit(2)
    with open(PKL, "rb") as f:
        obj = InterceptUnpickler(f).load()

    boosters = []; walk(obj, boosters)
    if not boosters:
        print("[FAIL] no boosters found"); sys.exit(1)

    outdir = Path("extracted_boosters"); outdir.mkdir(exist_ok=True)
    count = 0
    for i, b in enumerate(boosters):
        raw = to_bytes(getattr(b, "state", None))
        if not raw: continue
        (outdir / f"booster_{i}.raw").write_bytes(raw)
        print(f"[INFO] wrote {outdir}/booster_{i}.raw ({len(raw)} bytes)")
        count += 1
    print(("[PASS] extracted %d booster(s)" % count) if count else "[FAIL] none")
    sys.exit(0 if count else 1)

if __name__ == "__main__":
    main()
