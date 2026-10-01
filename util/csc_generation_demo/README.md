# Client-side caching: invalidation generation demo

A self-contained, **native** demonstration and test harness for the client-side
caching (CSC) *invalidation generation*. It uses no Docker and no Redis binary:
`fake_redis_server.py` is a small in-process TCP server that speaks just enough
RESP3 (HELLO handshake advertised as redis 7.4.0, `CLIENT TRACKING ON`, `PING`,
`GET`, and server-initiated `invalidate` push messages) to drive the real
synchronous `redis.Redis` client.

## What it verifies

1. **read** – the first `GET` learns the value over the wire; the second is a
   cache hit (no new wire `GET`); the stored entry is stamped with the cache's
   current generation.
2. **disconnect / reconnect** – disconnecting a tracked connection flushes the
   cache and advances the *cache-wide* generation; reconnecting advances the
   *per-connection* generation and installs a new invalidation push handler.
3. **no cross-generation hit** – after reconnect the stale value can never be a
   hit; the next `GET` is re-learned on the new socket and stamped with the new
   generation.
4. **out-of-order invalidation** – an invalidation produced by the *previous*
   socket but delivered late (replayed through the captured old handler after
   the reconnect) is dropped; both a key invalidation `[invalidate, [foo]]` and
   a whole-cache flush `[invalidate, null]` leave the fresh entry untouched.
5. **live invalidation still applies** – an invalidation genuinely delivered by
   the current connection invalidates the entry as before.

## Reproduce

From the repository root:

```sh
# Stand-alone script: prints each step, the wire GET count, cache size and
# generations; exits non-zero if any assertion fails:
PYTHONPATH=. python util/csc_generation_demo/demo_invalidation_generation.py

# The same scenario as a pytest test (plus a live flush test), native only:
python -m pytest tests/test_cache_generation_e2e.py -v
```

If unrelated pytest plugins in the environment are broken (`pytest-aiohttp`,
`pytest-httpbin`), they can be disabled; these tests need only the standard
library and redis-py itself:

```sh
python -m pytest tests/test_cache_generation_e2e.py -v \
    -p no:aiohttp -p no:asyncio -p no:httpbin
```

## Expected key results

```
first  GET foo -> b'v1'; wire GETs so far = 1
second GET foo -> b'v1' (cache hit); wire GETs = 1
after disconnect : cache size = 0; cache generation = 1; connection generation = 2
after reconnect  : connection generation = 3; fresh handler installed = True
GET foo -> b'v2' (re-learned); wire GETs = 2
replayed stale [invalidate, [foo]]  -> dropped; GET stays a cache hit
replayed stale [invalidate, null]   -> dropped
ALL CHECKS PASSED
```
