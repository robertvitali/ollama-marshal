"""Model size registry with background benchmarking and caching.

Tracks three pieces of per-model state:

1. **Loaded VRAM size** (`model_sizes.json`) — measured by loading the
   model and reading `/api/ps`. Used by the scheduler to decide if a
   model fits in remaining VRAM.
2. **Architecture metadata** (`model_metadata.json`) — extracted from
   `/api/show`. Captures `max_context_length`, `num_layers`,
   `kv_per_slot_at_max_ctx`. Used to (a) inject `options.num_ctx` so
   Ollama doesn't silently truncate context, and (b) compute how many
   concurrent inference slots fit per loaded model.
3. **Known-models set** (in-memory, sourced from `/api/tags`). Used by
   the server to fail-fast on requests for models that aren't installed
   in Ollama, avoiding the 1h request-timeout death-loop where marshal
   keeps trying to preload a model that doesn't exist.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog

from ollama_marshal.config import KVCacheType

logger = structlog.get_logger()

# KV cache size scales with quantization. f16 stores 2 bytes per element;
# q8_0 stores 1 byte (~50%); q4_0 stores ~0.5 bytes (~25%). These
# multipliers approximate the slot-size delta and are applied on top of
# the f16-derived ``kv_per_slot_at_max_ctx`` from /api/show.
_KV_PRECISION_MULTIPLIER: dict[KVCacheType, float] = {
    KVCacheType.F16: 1.0,
    KVCacheType.Q8_0: 0.5,
    KVCacheType.Q4_0: 0.25,
}

_DEFAULT_REGISTRY_PATH = Path.home() / ".ollama-marshal" / "model_sizes.json"
_DEFAULT_METADATA_PATH = Path.home() / ".ollama-marshal" / "model_metadata.json"

# KV cache dtype defaults to fp16 = 2 bytes per element. Newer Ollama
# supports KV cache quantization (q8_0 = 1 byte) but we conservatively
# assume the heaviest case so we don't over-allocate parallel slots.
_DEFAULT_KV_DTYPE_BYTES = 2

# Conservative fallback when /api/show doesn't expose architecture info
# (e.g. older Ollama, custom modelfile). Both values are intentionally
# small so the math stays safe — under-estimating context means we'd
# inject a small num_ctx (no harm, model still works), and under-
# estimating layers/dim means we'd compute a larger kv_per_slot than
# real (so we'd allow fewer parallel slots than actually fit — also
# safe, just less performant).
_FALLBACK_MAX_CONTEXT = 4096
_FALLBACK_NUM_LAYERS = 32
_FALLBACK_KV_DIM = 4096

# Cap on how often `is_known_model` may opportunistically re-sync the
# /api/tags list when asked about a model it doesn't recognize. Prevents
# a flood of unknown-model requests from DOSing Ollama's /api/tags
# endpoint while still letting marshal pick up newly-pulled models
# within a few seconds.
_KNOWN_MODELS_RESYNC_MIN_INTERVAL_S = 5.0


class MalformedTagsResponseError(Exception):
    """Raised when /api/tags returns a non-JSON or wrong-shape body.

    Distinguished from `httpx.HTTPError` so the registry sync path
    can treat a malformed response as a *failed* sync rather than as
    an authoritative "Ollama has zero models" — committing the empty
    set as truth would falsely prune `_known_models` AND delete the
    on-disk size/metadata caches on a single transient bad response
    (e.g. a proxy mid-flight returning a text/plain error page).
    Callers should catch this alongside `httpx.HTTPError` and skip
    state mutation.
    """


@dataclass
class ModelMetadata:
    """Architecture-derived metadata used for context + parallelism math.

    Fields are extracted from `/api/show`'s `model_info` block. The
    architecture prefix (e.g. `llama`, `qwen3`, `gemma3`) is read from
    `general.architecture` and used to find the right keys.

    The `kv_per_slot_at_max_ctx` field is computed once at probe time
    so consumers don't have to know the math:

        kv_per_slot = max_ctx * num_layers * kv_dim * dtype_bytes * 2
                                                                   ^^^
                                                  factor 2 for K and V

    where `kv_dim = (embedding_length / head_count) * head_count_kv`
    (head_dim multiplied by number of KV heads, accounting for grouped-
    query attention).
    """

    name: str
    architecture: str
    max_context_length: int
    num_layers: int
    embedding_length: int
    head_count: int
    head_count_kv: int
    kv_dtype_bytes: int = _DEFAULT_KV_DTYPE_BYTES
    probed_at: str = ""

    @property
    def head_dim(self) -> int:
        """Per-head dimension = embedding_length // head_count."""
        if self.head_count == 0:
            return 0
        return self.embedding_length // self.head_count

    @property
    def kv_dim(self) -> int:
        """Effective KV dimension = head_dim * head_count_kv (GQA-aware)."""
        return self.head_dim * self.head_count_kv

    @property
    def kv_per_slot_at_max_ctx(self) -> int:
        """Bytes of KV cache one inference slot needs at full context."""
        return self.kv_per_slot_at_ctx(self.max_context_length)

    def kv_per_slot_at_ctx(self, ctx: int) -> int:
        """Bytes of KV cache one inference slot needs at the given ctx length."""
        return ctx * self.num_layers * self.kv_dim * self.kv_dtype_bytes * 2

    def to_json_dict(self) -> dict[str, Any]:
        """Serializable dict for the on-disk cache."""
        d = asdict(self)
        # Include computed fields so external readers don't have to recompute.
        d["kv_per_slot_at_max_ctx"] = self.kv_per_slot_at_max_ctx
        return d

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Reconstruct from on-disk cache. Ignores computed fields."""
        return cls(
            name=data["name"],
            architecture=data["architecture"],
            max_context_length=int(data["max_context_length"]),
            num_layers=int(data["num_layers"]),
            embedding_length=int(data["embedding_length"]),
            head_count=int(data["head_count"]),
            head_count_kv=int(data["head_count_kv"]),
            kv_dtype_bytes=int(data.get("kv_dtype_bytes", _DEFAULT_KV_DTYPE_BYTES)),
            probed_at=str(data.get("probed_at", "")),
        )


@dataclass
class _ProbeResult:
    """Internal struct returned by `_probe_show`.

    Reports whether all fields were extracted cleanly or if any had to
    fall back to defaults. Callers use this to log appropriate warnings
    so silent fallback decisions are visible in the audit trail.
    """

    metadata: ModelMetadata
    used_fallback: bool = False
    missing_fields: list[str] = field(default_factory=list)


class ModelRegistry:
    """Tracks the VRAM size of each Ollama model.

    On startup, loads a cached registry from disk and diffs it against
    the current set of downloaded models. New models are benchmarked
    in the background. Deleted models are removed from the cache.

    Attributes:
        registry_path: Path to the JSON cache file.
        ollama_host: Base URL of the Ollama API.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        registry_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.ollama_host = ollama_host
        self.registry_path = registry_path or _DEFAULT_REGISTRY_PATH
        self.metadata_path = metadata_path or _DEFAULT_METADATA_PATH
        self._sizes: dict[str, int] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        self._benchmarking: set[str] = set()
        # Source-of-truth set for "is this model installed in Ollama right
        # now". Populated by _sync_with_ollama and refreshed
        # opportunistically by is_known_model(). Distinct from `_sizes`
        # (which only contains benchmarked models) and `_metadata` (probed).
        self._known_models: set[str] = set()
        self._known_models_last_sync: float = 0.0
        # Background polling task that periodically calls
        # _sync_with_ollama. Started by start_polling() in the server
        # lifespan; without it, _known_models would only refresh at
        # startup and on opportunistic miss in is_known_model().
        # Without periodic refresh, an `ollama rm <model>` while marshal
        # is running leaves the registry stale until restart, causing
        # subsequent requests for the removed model to preload-loop
        # into 502 instead of fail-fast 404.
        self._poll_task: asyncio.Task[None] | None = None
        self._poll_interval_s: float = 0.0
        # Serializes _sync_with_ollama against itself so concurrent
        # callers (the periodic poll loop + is_known_model's
        # opportunistic resync) can't race on out-of-order /api/tags
        # responses. Without the lock, two in-flight fetches could
        # commit snapshots in arbitrary order — the LATER fetch (with
        # newer state) might be overwritten by the EARLIER fetch's
        # response, falsely pruning a freshly-pulled model from
        # _known_models AND _sizes/_metadata on disk. The race is
        # cheap to trigger when poll_interval_s is short (integration
        # tests) and reachable in production whenever an unknown-model
        # request arrives during a periodic poll.
        self._sync_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load cached registry and sync with current Ollama models."""
        self._load_cache()
        self._load_metadata_cache()
        await self._sync_with_ollama()

    def _load_cache(self) -> None:
        """Load model sizes from the JSON cache file."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                if isinstance(data, dict):
                    self._sizes = {k: int(v) for k, v in data.items()}
                    logger.info(
                        "model_registry.cache_loaded", model_count=len(self._sizes)
                    )
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "model_registry.cache_corrupt", path=str(self.registry_path)
                )
                self._sizes = {}

    def _save_cache(self) -> None:
        """Write current model sizes to the JSON cache file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self._sizes, indent=2) + "\n")
        logger.debug("model_registry.cache_saved", model_count=len(self._sizes))

    def _load_metadata_cache(self) -> None:
        """Load architecture metadata from `model_metadata.json`.

        Mirrors `_load_cache`. On parse failure, log a warning and start
        with an empty metadata map — probes will repopulate on demand.
        """
        if not self.metadata_path.exists():
            return
        try:
            data = json.loads(self.metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "model_registry.metadata_cache_corrupt",
                path=str(self.metadata_path),
            )
            self._metadata = {}
            return
        if not isinstance(data, dict):
            self._metadata = {}
            return
        loaded: dict[str, ModelMetadata] = {}
        for name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            try:
                loaded[name] = ModelMetadata.from_json_dict(entry)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "model_registry.metadata_entry_skipped",
                    model=name,
                    error=str(exc),
                )
        self._metadata = loaded
        if loaded:
            logger.info("model_registry.metadata_cache_loaded", model_count=len(loaded))

    def _save_metadata_cache(self) -> None:
        """Write the metadata cache to disk."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {name: m.to_json_dict() for name, m in self._metadata.items()}
        self.metadata_path.write_text(json.dumps(serialized, indent=2) + "\n")
        logger.debug(
            "model_registry.metadata_cache_saved", model_count=len(self._metadata)
        )

    async def _sync_with_ollama(self) -> None:
        """Sync the registry against models currently downloaded in Ollama.

        Refreshes `_known_models` (the fail-fast lookup set used by
        `is_known_model`) and prunes stale entries from the on-disk
        caches. Called both at startup (from `initialize`) and on a
        background poll loop (`start_polling`) so model adds/removals
        in Ollama show up within one poll interval.

        Serialized via `_sync_lock` so the periodic poll can't race
        with `is_known_model`'s opportunistic resync — concurrent
        in-flight /api/tags fetches whose responses arrive out of
        order would otherwise commit older state on top of newer,
        falsely pruning freshly-pulled models from `_known_models`
        and the on-disk caches.

        Logging is delta-aware: the first sync logs the full inventory
        of un-benchmarked models (so startup output matches v0.6.5
        behavior), and subsequent syncs log only models that were
        added or removed since the previous sync. Without that, every
        poll cycle would re-emit the full un-benchmarked list.
        """
        async with self._sync_lock:
            try:
                current_models = await self._fetch_model_list()
            except httpx.HTTPError:
                logger.warning(
                    "model_registry.sync_failed", reason="cannot reach Ollama"
                )
                return
            except MalformedTagsResponseError as exc:
                # A 200 with garbage body must NOT be committed as an
                # authoritative empty inventory — that would wipe
                # _known_models and delete the on-disk size/metadata
                # caches on a single transient bad response. Treat
                # exactly like an httpx error: skip state mutation,
                # leave _known_models_last_sync unchanged so
                # is_known_model's fail-open detection still fires.
                logger.warning(
                    "model_registry.sync_failed",
                    reason="malformed /api/tags response",
                    detail=str(exc),
                )
                return

            new_set = set(current_models)
            is_first_sync = self._known_models_last_sync == 0.0
            added = new_set - self._known_models
            removed = self._known_models - new_set

            # Refresh the source-of-truth set used by is_known_model().
            self._known_models = new_set
            self._known_models_last_sync = time.monotonic()

            # Remove entries for deleted models (from both on-disk caches).
            stale = set(self._sizes.keys()) - new_set
            meta_stale = set(self._metadata.keys()) - new_set
            for model in stale:
                del self._sizes[model]
            for model in meta_stale:
                del self._metadata[model]

            if stale:
                self._save_cache()
            if meta_stale:
                self._save_metadata_cache()

            if is_first_sync:
                unknown = new_set - set(self._sizes.keys())
                if unknown:
                    logger.info(
                        "model_registry.new_models_found", models=sorted(unknown)
                    )
                # Pre-Bug-6, the per-model `removed_stale` log fired here.
                # The delta-aware refactor accidentally muted it for the
                # startup path — operators tuning a fresh marshal lose
                # visibility into "we deleted these from disk because
                # they're no longer in Ollama". Restore an aggregate
                # version so the audit trail stays complete.
                if stale or meta_stale:
                    logger.info(
                        "model_registry.removed_stale_at_startup",
                        models=sorted(stale | meta_stale),
                    )
            else:
                if added:
                    logger.info("model_registry.models_added", models=sorted(added))
                if removed:
                    logger.info("model_registry.models_removed", models=sorted(removed))

    async def start_polling(self, poll_interval_s: float) -> None:
        """Begin periodic /api/tags polling so model adds/removals show up live.

        Called from the server lifespan after `initialize`. The poll
        loop fires `_sync_with_ollama` every `poll_interval_s` seconds.
        Calling twice without `stop_polling` between is a no-op (the
        existing task wins).

        Args:
            poll_interval_s: Seconds between /api/tags polls. Pulled
                from `config.scheduler.model_detect_interval` in the
                production wiring; tests pass smaller values to keep
                wall-clock low.
        """
        if self._poll_task is not None and not self._poll_task.done():
            return
        self._poll_interval_s = poll_interval_s
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("model_registry.polling_started", interval_s=poll_interval_s)

    async def stop_polling(self) -> None:
        """Stop the background polling task. Idempotent.

        Catches any exception from the awaited task (not just
        `CancelledError`) so a poll task that died with an unforeseen
        error class doesn't break the surrounding `lifespan` finally
        block — which would otherwise skip subsequent shutdown steps
        like `unload_models`.
        """
        if self._poll_task is None:
            return
        self._poll_task.cancel()
        try:
            await self._poll_task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Catch (don't swallow silently) so an unforeseen poll
            # task failure doesn't break the surrounding lifespan
            # finally and skip subsequent shutdown steps.
            logger.warning("model_registry.poll_task_died", exc_info=True)
        self._poll_task = None
        logger.info("model_registry.polling_stopped")

    async def _poll_loop(self) -> None:
        """Continuously call `_sync_with_ollama` at the configured interval.

        `_sync_with_ollama` already swallows `httpx.HTTPError`. The
        outer `except Exception` is a defense-in-depth net for
        anything else (e.g. `OSError` from `_save_cache` on a full
        disk) so one bad poll doesn't kill the loop and silently
        freeze the known-models view until restart. `CancelledError`
        is a `BaseException` and falls through naturally — that's how
        `stop_polling` ends the task.
        """
        while True:
            try:
                await asyncio.sleep(self._poll_interval_s)
                await self._sync_with_ollama()
            except Exception:
                logger.warning("model_registry.poll_error", exc_info=True)

    async def fetch_model_list(self) -> list[str]:
        """Fetch the list of downloaded model names from Ollama (public API).

        Used by external modules that need a fresh /api/tags read
        (e.g. the `marshal doctor` CLI). Internal callers also use this
        method — `_fetch_model_list` is preserved as a thin alias for
        backwards compatibility with existing tests.

        Top-level shape errors (non-JSON body, response not a dict,
        ``"models"`` not a list) raise `MalformedTagsResponseError`
        rather than returning an empty list. Returning `[]` would be
        indistinguishable from "Ollama has zero models installed" —
        and the caller (`_sync_with_ollama`) would commit that empty
        set as authoritative truth, wiping `_known_models` AND
        deleting cached benchmarks/metadata on disk on a single
        transient bad response (e.g. a proxy mid-flight returning a
        text/plain error page). Per-entry malformations (missing
        ``"name"``, wrong type) are still tolerated: that entry is
        dropped and the rest are returned, since the document IS
        valid Ollama-shaped data — just one bad row.

        Returns:
            List of model name strings.

        Raises:
            httpx.HTTPError: Connection failed, timed out, or non-2xx.
            MalformedTagsResponseError: 200 response with body that
                doesn't match the expected ``{"models": [...]}`` shape.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.ollama_host}/api/tags", timeout=10)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as exc:
                # JSONDecodeError is a subclass of ValueError. Surface
                # as MalformedTagsResponseError so the sync path can
                # fail-soft (don't update state) rather than treat the
                # garbled body as an authoritative empty inventory.
                raise MalformedTagsResponseError(
                    f"/api/tags returned non-JSON body from {self.ollama_host}"
                ) from exc
        if not isinstance(data, dict):
            raise MalformedTagsResponseError(
                f"/api/tags returned {type(data).__name__}, expected dict"
            )
        models_raw = data.get("models", [])
        if not isinstance(models_raw, list):
            raise MalformedTagsResponseError(
                "/api/tags returned 'models' that is not a list"
            )
        out: list[str] = []
        for m in models_raw:
            if isinstance(m, dict):
                name = m.get("name")
                if isinstance(name, str) and name:
                    out.append(name)
        return out

    # Kept for backwards compatibility with existing tests that patch
    # `_fetch_model_list`. New callers should use the public name.
    _fetch_model_list = fetch_model_list

    async def benchmark_model(self, model: str) -> int | None:
        """Benchmark a single model by loading it and measuring VRAM.

        Loads the model with an empty prompt, reads size_vram from /api/ps,
        then unloads it. Result is cached.

        Args:
            model: The model name to benchmark.

        Returns:
            VRAM size in bytes, or None if benchmarking failed.
        """
        if model in self._benchmarking:
            return None
        self._benchmarking.add(model)
        logger.info("model_registry.benchmarking", model=model)

        try:
            async with httpx.AsyncClient() as client:
                # Load the model
                await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": "5m"},
                    timeout=300,
                )

                # Read its VRAM size
                resp = await client.get(f"{self.ollama_host}/api/ps", timeout=10)
                resp.raise_for_status()
                ps_data = resp.json()

                size_vram = self._extract_model_vram(ps_data, model)
                if size_vram is not None:
                    self._sizes[model] = size_vram
                    self._save_cache()
                    logger.info(
                        "model_registry.benchmarked",
                        model=model,
                        size_vram=size_vram,
                        size_gb=round(size_vram / (1024**3), 2),
                    )

                # Unload the model
                await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": "0"},
                    timeout=60,
                )

                return size_vram

        except httpx.HTTPError:
            logger.warning("model_registry.benchmark_failed", model=model)
            return None
        finally:
            self._benchmarking.discard(model)

    async def benchmark_unknown(self) -> None:
        """Benchmark sizes + probe metadata for all models not yet cached.

        Loads each unknown model one at a time to avoid VRAM conflicts.
        Metadata probing is much cheaper (just `/api/show`, no actual
        model load) so we do it for all models that need it, not just
        the size-unknown set.
        """
        try:
            current_models = await self._fetch_model_list()
        except (httpx.HTTPError, MalformedTagsResponseError):
            return

        # Metadata probe is cheap — fan out for any model missing metadata,
        # whether or not its size is also unknown.
        meta_unknown = [m for m in current_models if m not in self._metadata]
        if meta_unknown:
            logger.info(
                "model_registry.metadata_probe_starting", count=len(meta_unknown)
            )
            for model in meta_unknown:
                await self.probe_metadata(model)

        size_unknown = [m for m in current_models if m not in self._sizes]
        if not size_unknown:
            logger.info("model_registry.all_benchmarked")
            return

        logger.info("model_registry.benchmark_starting", count=len(size_unknown))
        for model in size_unknown:
            await self.benchmark_model(model)

    def get_model_size(self, model: str) -> int | None:
        """Get the cached VRAM size for a model.

        Args:
            model: The model name.

        Returns:
            VRAM size in bytes, or None if not yet benchmarked.
        """
        return self._sizes.get(model)

    def is_benchmarked(self, model: str) -> bool:
        """Check if a model has been benchmarked.

        Args:
            model: The model name.

        Returns:
            True if the model's VRAM size is known.
        """
        return model in self._sizes

    def list_models(self) -> dict[str, int]:
        """Get all cached model sizes.

        Returns:
            Dict mapping model names to VRAM sizes in bytes.
        """
        return dict(self._sizes)

    def remove_model(self, model: str) -> None:
        """Remove a model from the registry (both size and metadata caches).

        Args:
            model: The model name to remove.
        """
        size_changed = model in self._sizes
        meta_changed = model in self._metadata
        if size_changed:
            del self._sizes[model]
        if meta_changed:
            del self._metadata[model]
        if size_changed:
            self._save_cache()
        if meta_changed:
            self._save_metadata_cache()

    async def probe_metadata(self, model: str) -> ModelMetadata | None:
        """Fetch architecture metadata from `/api/show` and cache it.

        Idempotent — if `model` is already in the metadata cache, returns
        the cached entry without re-probing.

        Returns:
            ModelMetadata for the model, or None if Ollama was unreachable.
            On partial / unknown architecture, returns metadata with
            fallback fields filled in (logs a warning).
        """
        cached = self._metadata.get(model)
        if cached is not None:
            return cached
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_host}/api/show",
                    json={"name": model},
                    timeout=10,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "model_registry.metadata_probe_failed",
                model=model,
                error=str(exc) or repr(exc),
                error_type=type(exc).__name__,
            )
            return None

        result = self._parse_show_response(model, data)
        self._metadata[model] = result.metadata
        self._save_metadata_cache()
        if result.used_fallback:
            logger.warning(
                "model_registry.metadata_fallback_used",
                model=model,
                missing_fields=result.missing_fields,
            )
        else:
            logger.info(
                "model_registry.metadata_probed",
                model=model,
                architecture=result.metadata.architecture,
                max_ctx=result.metadata.max_context_length,
                kv_per_slot_mb=round(
                    result.metadata.kv_per_slot_at_max_ctx / (1024**2), 1
                ),
            )
        return result.metadata

    @staticmethod
    def _parse_show_response(model: str, data: dict[str, Any]) -> _ProbeResult:
        """Extract architecture metadata from a parsed `/api/show` response.

        Ollama stores architecture-specific keys under `model_info` with a
        prefix derived from `general.architecture` (e.g. `llama.context_
        length`, `qwen3.context_length`). We read the architecture name
        first, then look up the per-arch keys.

        Returns a `_ProbeResult` with `used_fallback=True` when one or
        more fields had to be filled with a conservative default.
        """
        info = data.get("model_info", {}) or {}
        arch = str(info.get("general.architecture", "") or "unknown")

        def _get(key: str, default: int) -> tuple[int, bool]:
            """Return (value, used_fallback). Looks up `<arch>.<key>` from info."""
            full = f"{arch}.{key}"
            raw = info.get(full)
            if raw is None:
                return default, True
            try:
                return int(raw), False
            except (TypeError, ValueError):
                return default, True

        max_ctx, fb1 = _get("context_length", _FALLBACK_MAX_CONTEXT)
        n_layers, fb2 = _get("block_count", _FALLBACK_NUM_LAYERS)
        embed_len, fb3 = _get("embedding_length", _FALLBACK_KV_DIM)
        head_count, fb4 = _get("attention.head_count", 32)
        # head_count_kv defaults to head_count when not specified (no GQA).
        hck_raw = info.get(f"{arch}.attention.head_count_kv")
        if hck_raw is None:
            head_count_kv, fb5 = head_count, False
        else:
            try:
                head_count_kv, fb5 = int(hck_raw), False
            except (TypeError, ValueError):
                head_count_kv, fb5 = head_count, True

        used_fallback = any([fb1, fb2, fb3, fb4, fb5])
        missing = [
            k
            for k, fb in [
                ("context_length", fb1),
                ("block_count", fb2),
                ("embedding_length", fb3),
                ("attention.head_count", fb4),
                ("attention.head_count_kv", fb5),
            ]
            if fb
        ]

        meta = ModelMetadata(
            name=model,
            architecture=arch,
            max_context_length=max_ctx,
            num_layers=n_layers,
            embedding_length=embed_len,
            head_count=head_count,
            head_count_kv=head_count_kv,
            kv_dtype_bytes=_DEFAULT_KV_DTYPE_BYTES,
            probed_at=datetime.now(UTC).isoformat(),
        )
        return _ProbeResult(
            metadata=meta, used_fallback=used_fallback, missing_fields=missing
        )

    async def is_known_model(self, model: str) -> bool:
        """Return True if `model` is currently installed in Ollama.

        Used by the server to fail-fast on inference requests for models
        Ollama doesn't have, instead of letting `lifecycle.preload`
        retry for an hour until the request times out.

        Behavior:
        - If the model is in the cached `_known_models` set, return True
          immediately (zero HTTP).
        - If not in the cache AND it's been more than
          `_KNOWN_MODELS_RESYNC_MIN_INTERVAL_S` since the last sync,
          delegate to `_sync_with_ollama` (lock-serialized against the
          periodic poll) to refresh state. Catches the case where
          a user just ran `ollama pull <model>` and immediately fired
          a request — without this re-sync, we'd false-fail until the
          next periodic sync.
        - If `_sync_with_ollama` failed to reach Ollama (detectable via
          unchanged `_known_models_last_sync`), fail OPEN — better to
          attempt the preload and let Ollama answer than to wrongly
          404 on a transient hiccup.
        """
        if model in self._known_models:
            return True
        # Rate-limited opportunistic re-sync.
        now = time.monotonic()
        if now - self._known_models_last_sync < _KNOWN_MODELS_RESYNC_MIN_INTERVAL_S:
            return False  # too soon to re-sync; trust the cached negative
        # Delegate to the lock-serialized sync so we can't race the
        # periodic poll loop. The lock also coalesces concurrent
        # opportunistic resyncs from multiple in-flight requests for
        # the same unknown model.
        pre_sync_marker = self._known_models_last_sync
        await self._sync_with_ollama()
        if self._known_models_last_sync == pre_sync_marker:
            # Sync didn't advance — Ollama unreachable (or another
            # concurrent caller's sync also failed). Fail-open.
            logger.warning(
                "model_registry.known_models_resync_failed",
                model=model,
            )
            return True
        return model in self._known_models

    def get_metadata(self, model: str) -> ModelMetadata | None:
        """Return cached metadata for a model (None if not yet probed)."""
        return self._metadata.get(model)

    def get_max_context(self, model: str) -> int | None:
        """Convenience: max context length from cached metadata."""
        meta = self._metadata.get(model)
        return meta.max_context_length if meta is not None else None

    def get_kv_per_slot(self, model: str, ctx: int | None = None) -> int | None:
        """KV cache bytes per inference slot for a model at the given ctx.

        Args:
            model: Model name (must already have been probed).
            ctx: Context length to compute for. Defaults to the model's
                max context length.

        Returns:
            Bytes per slot, or None if metadata not yet cached.
        """
        meta = self._metadata.get(model)
        if meta is None:
            return None
        return meta.kv_per_slot_at_ctx(
            ctx if ctx is not None else meta.max_context_length
        )

    def get_kv_per_slot_scaled(
        self,
        model: str,
        kv_cache_type: KVCacheType,
        ctx: int | None = None,
    ) -> int | None:
        """KV cache bytes per slot adjusted for instance precision.

        The architectural ``kv_per_slot`` from /api/show assumes the f16
        default (2 bytes/element). When an instance is launched with a
        smaller ``OLLAMA_KV_CACHE_TYPE`` (q8_0 or q4_0), the slot uses
        proportionally less VRAM. This helper applies the precision
        multiplier so routing's fit math reflects the real cost on each
        instance.

        Args:
            model: Model name (must already have been probed).
            kv_cache_type: Precision of the target instance.
            ctx: Context length to compute for. Defaults to model max.

        Returns:
            Scaled bytes per slot, or None if metadata not yet cached.
        """
        base = self.get_kv_per_slot(model, ctx=ctx)
        if base is None:
            return None
        multiplier = _KV_PRECISION_MULTIPLIER[kv_cache_type]
        return int(base * multiplier)

    async def get_total_footprint(
        self,
        model: str,
        num_ctx: int,
        kv_cache_type: KVCacheType,
    ) -> int:
        """Estimate total VRAM footprint of `model` at `num_ctx` on an instance.

        Footprint = model weights + KV slot at this context length, with
        the slot scaled to the instance's precision. Used by routing's
        ``probe_fit`` to decide whether a request would actually fit on a
        given instance without evicting work.

        Args:
            model: Model name.
            num_ctx: Allocated context length for the slot.
            kv_cache_type: Precision of the target instance.

        Returns:
            Estimated bytes. Falls back to model size + 0 KV when
            metadata isn't cached (conservative — better to under-
            estimate KV than refuse the request).
        """
        size = await self.get_or_estimate_size(model)
        kv = self.get_kv_per_slot_scaled(model, kv_cache_type, ctx=num_ctx) or 0
        return size + kv

    async def get_or_estimate_size(self, model: str) -> int:
        """Get model size from cache, or estimate from /api/show.

        Falls back to the model's file size from /api/show as a rough
        estimate when the model hasn't been benchmarked yet.

        Args:
            model: The model name.

        Returns:
            VRAM size in bytes (exact if benchmarked, estimated otherwise).
        """
        cached = self._sizes.get(model)
        if cached is not None:
            return cached

        # Estimate from /api/show
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_host}/api/show",
                    json={"name": model},
                    timeout=10,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                # Use model_info parameter count as rough estimate
                # Approximate: 4 bytes per parameter for Q4 quantization
                model_info = data.get("model_info", {})
                param_count = model_info.get("general.parameter_count", 0)
                if param_count:
                    estimated = int(param_count * 4)  # Q4 ~ 4 bytes/param
                    logger.debug(
                        "model_registry.estimated_size",
                        model=model,
                        estimated_bytes=estimated,
                    )
                    return estimated
        except httpx.HTTPError:
            pass

        # Last resort: return a conservative 4GB estimate
        logger.warning("model_registry.size_unknown", model=model, default="4GB")
        return 4 * 1024**3

    @staticmethod
    def _extract_model_vram(ps_data: dict[str, Any], model: str) -> int | None:
        """Extract size_vram for a model from /api/ps response data.

        Args:
            ps_data: The parsed JSON response from /api/ps.
            model: The model name to look for.

        Returns:
            VRAM size in bytes, or None if model not found.
        """
        for m in ps_data.get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return int(m.get("size_vram", 0))
        return None
