from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fusion_core.free_pool import (
    build_free_panel_spec,
    discover_free_models,
    normalize_free_models,
    resolve_free_panel,
    select_diverse_models,
)


class FreePoolTests(unittest.TestCase):
    def test_normalize_filters_paid_nontext_and_short_context(self) -> None:
        payload = {
            "data": [
                {
                    "id": "alpha/reasoner:free",
                    "name": "Reasoner",
                    "context_length": 131072,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"output_modalities": ["text"]},
                    "supported_parameters": ["reasoning", "response_format"],
                },
                {
                    "id": "beta/paid",
                    "context_length": 1000000,
                    "pricing": {"prompt": "0.1", "completion": "0.2"},
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "gamma/embedding:free",
                    "context_length": 100000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"output_modalities": ["embedding"]},
                },
                {
                    "id": "delta/tiny:free",
                    "context_length": 4096,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "openrouter/free",
                    "context_length": 200000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"output_modalities": ["text"]},
                },
            ]
        }
        rows = normalize_free_models(payload, min_context=32000)
        self.assertEqual([row["id"] for row in rows], ["alpha/reasoner:free"])
        self.assertEqual(rows[0]["provider_family"], "alpha")
        self.assertGreater(rows[0]["capability_score"], 0)

    def test_diversity_prefers_distinct_provider_families(self) -> None:
        rows = [
            {"id": "a/one:free", "provider_family": "a"},
            {"id": "a/two:free", "provider_family": "a"},
            {"id": "b/one:free", "provider_family": "b"},
            {"id": "c/one:free", "provider_family": "c"},
        ]
        chosen = select_diverse_models(rows, size=3)
        self.assertEqual([row["id"] for row in chosen], ["a/one:free", "b/one:free", "c/one:free"])

    def test_panel_assigns_roles_without_losing_model_ids(self) -> None:
        rows = [
            {"id": "a/one:free", "provider_family": "a"},
            {"id": "b/two:free", "provider_family": "b"},
            {"id": "c/three:free", "provider_family": "c"},
        ]
        panel = build_free_panel_spec(rows, size=3)
        self.assertEqual(
            panel,
            "or:a/one:free@builder,or:b/two:free@expert,or:c/three:free@falsifier",
        )

    def test_discovery_uses_fresh_cache_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps({
                "fetched_at": 99999999999,
                "models": [{"id": "cached/model:free", "provider_family": "cached"}],
            }))

            def fail(_timeout):
                raise AssertionError("network should not be called")

            rows, source = discover_free_models(cache_path=path, request_fn=fail)
            self.assertEqual(source, "cache")
            self.assertEqual(rows[0]["id"], "cached/model:free")

    def test_discovery_falls_back_to_stale_cache_on_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps({
                "fetched_at": 0,
                "models": [{"id": "stale/model:free", "provider_family": "stale"}],
            }))

            def fail(_timeout):
                raise OSError("offline")

            rows, source = discover_free_models(cache_path=path, cache_ttl=1, request_fn=fail)
            self.assertEqual(source, "stale-cache")
            self.assertEqual(rows[0]["id"], "stale/model:free")

    def test_resolve_returns_metadata_and_deterministic_panel(self) -> None:
        payload = {
            "data": [
                {
                    "id": "a/one:free",
                    "context_length": 100000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"output_modalities": ["text"]},
                    "supported_parameters": ["reasoning"],
                },
                {
                    "id": "b/two:free",
                    "context_length": 80000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"output_modalities": ["text"]},
                    "supported_parameters": [],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            panel, meta = resolve_free_panel(
                size=2,
                cache_path=Path(directory) / "cache.json",
                refresh=True,
                request_fn=lambda _timeout: payload,
            )
        self.assertEqual(meta["source"], "openrouter")
        self.assertEqual(meta["available"], 2)
        self.assertIn("or:a/one:free@builder", panel)
        self.assertIn("or:b/two:free@expert", panel)


if __name__ == "__main__":
    unittest.main()
