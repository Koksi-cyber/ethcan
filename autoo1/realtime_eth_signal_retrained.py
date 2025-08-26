"""
Realtime signal generator using the retrained model.

This script is identical to `realtime_eth_signal.py` but defaults to
loading `eth_signal_model_retrained.pkl`, the updated model trained on
the recent six‑month dataset.  It fetches recent 1‑minute candles from
Binance, computes engineered features, scores the latest candle with
the model and emits a long signal when the probability exceeds the
stored or user‑supplied threshold.

See realtime_eth_signal.py for full usage details.
"""

from realtime_eth_signal import main as _main


def main(argv=None):
    """Entry point that sets default model to the retrained one."""
    import argparse
    # Insert the default model path into argv if --model is not already
    import sys
    args_list = list(argv or sys.argv[1:])
    if '--model' not in args_list:
        args_list += ['--model', 'eth_signal_model_retrained.pkl']
    return _main(args_list)


if __name__ == '__main__':
    import sys as _sys
    raise SystemExit(main(_sys.argv[1:]))