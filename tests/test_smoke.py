from battery_strategy.planning import build_company_queries, build_market_queries
from battery_strategy.settings import load_manifest, load_runtime_config


def test_runtime_config_loads() -> None:
    cfg = load_runtime_config('configs/runtime.example.yaml')
    manifest = load_manifest(cfg.manifest_path)
    assert cfg.project_name == 'battery_strategy_multi_agent'
    assert len(manifest.sources) == 3


def test_queries_are_annotated() -> None:
    market_queries = build_market_queries('goal')
    company_queries = build_company_queries('LGES', 'goal')
    assert all('|' in item for item in market_queries)
    assert all('|' in item for item in company_queries)
