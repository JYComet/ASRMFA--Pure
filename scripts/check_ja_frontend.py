"""Read-only frontend capability and contract checker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .ja_en_schema import JAContractError, validate_config
    from .ja_frontend import FrontendConfig, probe_provider
    from .run_ja_en_pipeline import _load_yaml
except ImportError:  # direct script execution
    from ja_en_schema import JAContractError, validate_config
    from ja_frontend import FrontendConfig, probe_provider
    from run_ja_en_pipeline import _load_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check configured pyopenjtalk-plus frontend without writing a workspace")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-write", action="store_true", help="kept for explicit read-only invocation")
    args = parser.parse_args(argv)
    try:
        config = _load_yaml(Path(args.config).expanduser().absolute())
        validate_config(config, config_path=Path(args.config).expanduser().absolute())
        settings = FrontendConfig.from_mapping(config)
        probe = probe_provider(settings)
        print(json.dumps({"status": "OK", "read_only": True, "frontend": settings.identity(), "probe": probe}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except JAContractError as exc:
        print(json.dumps({"status": "BLOCKED", "error": exc.as_dict()}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
