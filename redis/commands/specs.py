"""
Registered command specifications for offline pipeline inspection.

This module owns the static registry that maps a Redis command name to the
information an offline inspector needs without contacting a server:

* the key positions (which argument slots hold key names), and
* the sensitive parameter slots (which argument slots hold secrets that must
  be redacted from any report).

The registry is deliberately small and conservative. Only commands whose key
positions are stable and well-known are registered; every other command is
reported as *unknown* rather than guessing that every string argument is a key.
This matches the Redis convention of ``first_key_pos`` / ``key_step``: a
command either declares its key slots explicitly or it does not take keys at
all.

Key positions are 1-indexed against the full argument list, where slot ``1``
is the first argument after the command name (``args[0]``). This mirrors the
Redis ``COMMAND`` reply's ``first_key_pos`` field.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

__all__ = [
    "CommandSpec",
    "InspectedCommand",
    "SensitiveParam",
    "SensitiveValue",
    "get_command_spec",
]


@dataclass(frozen=True, slots=True)
class SensitiveValue:
    """
    Placeholder for a redacted sensitive argument.

    The original value is never retained; only its :attr:`category` is
    reported, so a password can be identified as a password without leaking
    the secret itself.
    """

    category: str

    def __repr__(self) -> str:
        return f"<{self.category}>"


@dataclass(frozen=True, slots=True)
class SensitiveParam:
    """
    Declares that an argument slot holds a sensitive value.

    Attributes:
        position: 1-indexed position of the sensitive argument (slot ``1`` is
            the first argument after the command name).
        category: a label for the kind of secret, e.g. ``"password"``.
        when: an optional predicate over the full argument tuple; when given,
            the slot is redacted only if the predicate returns True. This lets
            a single entry cover a command whose sensitive slot depends on a
            sub-command (e.g. ``CONFIG SET requirepass``).
    """

    position: int
    category: str
    when: Optional[Callable[[Tuple], bool]] = None


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """
    Registered key and sensitive-parameter specification for one command.

    Attributes:
        first_key_pos: 1-indexed position of the first key argument. ``0``
            means the command takes no keys.
        key_step: step between consecutive key slots (``1`` for commands like
            ``GET``, ``2`` for ``MSET``).
        key_count: number of key slots when fixed; ``None`` for variadic
            commands whose key count depends on the arguments supplied.
        sensitive_params: slots that hold secrets.
    """

    first_key_pos: int = 0
    key_step: int = 1
    key_count: Optional[int] = None
    sensitive_params: Tuple[SensitiveParam, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class InspectedCommand:
    """
    One entry in an offline pipeline inspection report.

    Attributes:
        index: original position of the command in the pipeline's
            ``command_stack``, or ``None`` for synthesized MULTI/EXEC boundary
            commands.
        command: the command name (``args[0]``).
        args: the full argument tuple, with sensitive slots replaced by
            :class:`SensitiveValue`.
        key_positions: 1-indexed key slots, or ``None`` when the command is not
            registered (unknown). An empty tuple means the command is registered
            but takes no keys.
        in_transaction: whether the command executes inside the MULTI/EXEC
            transaction block.
        is_boundary: ``True`` for the synthesized MULTI and EXEC entries that
            mark the transaction boundaries.
    """

    index: Optional[int]
    command: str
    args: Tuple
    key_positions: Optional[Tuple[int, ...]]
    in_transaction: bool
    is_boundary: bool


def _redact_sensitive(args: Tuple, spec: CommandSpec) -> Tuple:
    """
    Return a copy of ``args`` with sensitive slots replaced by
    :class:`SensitiveValue`.

    The original tuple is never mutated; a new tuple is returned only when at
    least one slot is redacted, otherwise ``args`` is returned unchanged.

    Sensitive positions are 1-indexed against the full argument list, where
    slot ``1`` is the first argument after the command name (``args[0]``).
    """

    if not spec.sensitive_params:
        return args
    redacted = list(args)
    changed = False
    for param in spec.sensitive_params:
        idx = param.position
        if idx < 0 or idx >= len(redacted):
            continue
        if param.when is not None and not param.when(args):
            continue
        redacted[idx] = SensitiveValue(param.category)
        changed = True
    return tuple(redacted) if changed else args


def _compute_key_positions(spec: CommandSpec, args: Tuple) -> Tuple[int, ...]:
    """
    Compute the 1-indexed key slots for a registered command.

    For fixed-key commands the count is taken from the spec; for variadic
    commands slots are walked until the argument list is exhausted.
    """

    if spec.first_key_pos <= 0:
        return ()
    positions = []
    pos = spec.first_key_pos
    if spec.key_count is not None:
        for _ in range(spec.key_count):
            if pos >= len(args):
                break
            positions.append(pos)
            pos += spec.key_step
    else:
        while pos < len(args):
            positions.append(pos)
            pos += spec.key_step
    return tuple(positions)


def inspect_command(
    index: Optional[int],
    args: Tuple,
    in_transaction: bool,
    is_boundary: bool = False,
) -> InspectedCommand:
    """
    Build an :class:`InspectedCommand` for a single staged command.

    This is a pure, offline operation: it reads the registered spec and the
    supplied arguments only, never touches a connection, never encodes, and
    never mutates the pipeline's ``command_stack``.
    """

    command = args[0] if args else ""
    spec = get_command_spec(command)
    if spec is None:
        return InspectedCommand(
            index=index,
            command=command,
            args=args,
            key_positions=None,
            in_transaction=in_transaction,
            is_boundary=is_boundary,
        )
    redacted_args = _redact_sensitive(args, spec)
    key_positions = _compute_key_positions(spec, args)
    return InspectedCommand(
        index=index,
        command=command,
        args=redacted_args,
        key_positions=key_positions,
        in_transaction=in_transaction,
        is_boundary=is_boundary,
    )


# ---------------------------------------------------------------------------
# Static command registry
#
# Key positions follow the Redis ``COMMAND`` reply convention: ``first_key_pos``
# is 1-indexed (slot 1 is the first argument after the command name),
# ``key_step`` is the stride between keys, and ``key_count`` is ``None`` for
# variadic commands.
# ---------------------------------------------------------------------------

_AUTH_PASSWORD = SensitiveParam(position=1, category="password")
_CONFIG_SET_PASSWORD = SensitiveParam(
    position=3,
    category="password",
    when=lambda args: (
        len(args) >= 3
        and str(args[1]).lower() == "set"
        and str(args[2]).lower() == "requirepass"
    ),
)

COMMAND_SPECS: Dict[str, CommandSpec] = {
    # --- keyed commands ---------------------------------------------------
    "get": CommandSpec(first_key_pos=1, key_count=1),
    "set": CommandSpec(first_key_pos=1, key_count=1),
    "setnx": CommandSpec(first_key_pos=1, key_count=1),
    "setex": CommandSpec(first_key_pos=1, key_count=1),
    "psetex": CommandSpec(first_key_pos=1, key_count=1),
    "getset": CommandSpec(first_key_pos=1, key_count=1),
    "getdel": CommandSpec(first_key_pos=1, key_count=1),
    "getex": CommandSpec(first_key_pos=1, key_count=1),
    "append": CommandSpec(first_key_pos=1, key_count=1),
    "strlen": CommandSpec(first_key_pos=1, key_count=1),
    "getrange": CommandSpec(first_key_pos=1, key_count=1),
    "setrange": CommandSpec(first_key_pos=1, key_count=1),
    "substr": CommandSpec(first_key_pos=1, key_count=1),
    "incr": CommandSpec(first_key_pos=1, key_count=1),
    "decr": CommandSpec(first_key_pos=1, key_count=1),
    "incrby": CommandSpec(first_key_pos=1, key_count=1),
    "decrby": CommandSpec(first_key_pos=1, key_count=1),
    "incrbyfloat": CommandSpec(first_key_pos=1, key_count=1),
    "exists": CommandSpec(first_key_pos=1),
    "del": CommandSpec(first_key_pos=1),
    "unlink": CommandSpec(first_key_pos=1),
    "type": CommandSpec(first_key_pos=1, key_count=1),
    "expire": CommandSpec(first_key_pos=1, key_count=1),
    "expireat": CommandSpec(first_key_pos=1, key_count=1),
    "pexpire": CommandSpec(first_key_pos=1, key_count=1),
    "pexpireat": CommandSpec(first_key_pos=1, key_count=1),
    "persist": CommandSpec(first_key_pos=1, key_count=1),
    "ttl": CommandSpec(first_key_pos=1, key_count=1),
    "pttl": CommandSpec(first_key_pos=1, key_count=1),
    "dump": CommandSpec(first_key_pos=1, key_count=1),
    "restore": CommandSpec(first_key_pos=1, key_count=1),
    "restore-asking": CommandSpec(first_key_pos=1, key_count=1),
    "rename": CommandSpec(first_key_pos=1, key_count=2),
    "renamenx": CommandSpec(first_key_pos=1, key_count=2),
    "copy": CommandSpec(first_key_pos=1, key_count=2),
    "move": CommandSpec(first_key_pos=1, key_count=1),
    "object": CommandSpec(first_key_pos=2),
    "memory": CommandSpec(first_key_pos=2),
    "touch": CommandSpec(first_key_pos=1),
    "wait": CommandSpec(first_key_pos=0),
    "waitaof": CommandSpec(first_key_pos=0),
    # --- multi-key commands -----------------------------------------------
    "mget": CommandSpec(first_key_pos=1),
    "mset": CommandSpec(first_key_pos=1, key_step=2),
    "msetnx": CommandSpec(first_key_pos=1, key_step=2),
    "keys": CommandSpec(first_key_pos=0),
    "scan": CommandSpec(first_key_pos=0),
    "sort": CommandSpec(first_key_pos=1),
    "sort_ro": CommandSpec(first_key_pos=1),
    # --- hashes ------------------------------------------------------------
    "hget": CommandSpec(first_key_pos=1, key_count=1),
    "hset": CommandSpec(first_key_pos=1, key_count=1),
    "hsetnx": CommandSpec(first_key_pos=1, key_count=1),
    "hgetall": CommandSpec(first_key_pos=1, key_count=1),
    "hmget": CommandSpec(first_key_pos=1, key_count=1),
    "hmset": CommandSpec(first_key_pos=1, key_count=1),
    "hdel": CommandSpec(first_key_pos=1, key_count=1),
    "hexists": CommandSpec(first_key_pos=1, key_count=1),
    "hlen": CommandSpec(first_key_pos=1, key_count=1),
    "hkeys": CommandSpec(first_key_pos=1, key_count=1),
    "hvals": CommandSpec(first_key_pos=1, key_count=1),
    "hincrby": CommandSpec(first_key_pos=1, key_count=1),
    "hincrbyfloat": CommandSpec(first_key_pos=1, key_count=1),
    "hstrlen": CommandSpec(first_key_pos=1, key_count=1),
    "hrandfield": CommandSpec(first_key_pos=1, key_count=1),
    "hscan": CommandSpec(first_key_pos=1, key_count=1),
    # --- lists -------------------------------------------------------------
    "lrange": CommandSpec(first_key_pos=1, key_count=1),
    "lpush": CommandSpec(first_key_pos=1, key_count=1),
    "lpushx": CommandSpec(first_key_pos=1, key_count=1),
    "rpush": CommandSpec(first_key_pos=1, key_count=1),
    "rpushx": CommandSpec(first_key_pos=1, key_count=1),
    "lpop": CommandSpec(first_key_pos=1, key_count=1),
    "rpop": CommandSpec(first_key_pos=1, key_count=1),
    "blpop": CommandSpec(first_key_pos=1),
    "brpop": CommandSpec(first_key_pos=1),
    "llen": CommandSpec(first_key_pos=1, key_count=1),
    "lindex": CommandSpec(first_key_pos=1, key_count=1),
    "lset": CommandSpec(first_key_pos=1, key_count=1),
    "lrem": CommandSpec(first_key_pos=1, key_count=1),
    "ltrim": CommandSpec(first_key_pos=1, key_count=1),
    "linsert": CommandSpec(first_key_pos=1, key_count=1),
    "lpos": CommandSpec(first_key_pos=1, key_count=1),
    "lmove": CommandSpec(first_key_pos=1, key_count=2),
    "blmove": CommandSpec(first_key_pos=1, key_count=2),
    "rpoplpush": CommandSpec(first_key_pos=1, key_count=2),
    "brpoplpush": CommandSpec(first_key_pos=1, key_count=2),
    # --- sets --------------------------------------------------------------
    "sadd": CommandSpec(first_key_pos=1, key_count=1),
    "srem": CommandSpec(first_key_pos=1, key_count=1),
    "smembers": CommandSpec(first_key_pos=1, key_count=1),
    "sismember": CommandSpec(first_key_pos=1, key_count=1),
    "smismember": CommandSpec(first_key_pos=1, key_count=1),
    "scard": CommandSpec(first_key_pos=1, key_count=1),
    "srandmember": CommandSpec(first_key_pos=1, key_count=1),
    "spop": CommandSpec(first_key_pos=1, key_count=1),
    "sdiff": CommandSpec(first_key_pos=1),
    "sinter": CommandSpec(first_key_pos=1),
    "sunion": CommandSpec(first_key_pos=1),
    "sdiffstore": CommandSpec(first_key_pos=1, key_count=1),
    "sinterstore": CommandSpec(first_key_pos=1, key_count=1),
    "sunionstore": CommandSpec(first_key_pos=1, key_count=1),
    "smove": CommandSpec(first_key_pos=1, key_count=2),
    "sscan": CommandSpec(first_key_pos=1, key_count=1),
    # --- sorted sets -------------------------------------------------------
    "zadd": CommandSpec(first_key_pos=1, key_count=1),
    "zrem": CommandSpec(first_key_pos=1, key_count=1),
    "zrange": CommandSpec(first_key_pos=1, key_count=1),
    "zrevrange": CommandSpec(first_key_pos=1, key_count=1),
    "zrangebyscore": CommandSpec(first_key_pos=1, key_count=1),
    "zrevrangebyscore": CommandSpec(first_key_pos=1, key_count=1),
    "zrangebylex": CommandSpec(first_key_pos=1, key_count=1),
    "zrevrangebylex": CommandSpec(first_key_pos=1, key_count=1),
    "zrank": CommandSpec(first_key_pos=1, key_count=1),
    "zrevrank": CommandSpec(first_key_pos=1, key_count=1),
    "zscore": CommandSpec(first_key_pos=1, key_count=1),
    "zmscore": CommandSpec(first_key_pos=1, key_count=1),
    "zcard": CommandSpec(first_key_pos=1, key_count=1),
    "zcount": CommandSpec(first_key_pos=1, key_count=1),
    "zlexcount": CommandSpec(first_key_pos=1, key_count=1),
    "zincrby": CommandSpec(first_key_pos=1, key_count=1),
    "zpopmin": CommandSpec(first_key_pos=1, key_count=1),
    "zpopmax": CommandSpec(first_key_pos=1, key_count=1),
    "zmpop": CommandSpec(first_key_pos=1, key_count=1),
    "bzpopmin": CommandSpec(first_key_pos=1),
    "bzpopmax": CommandSpec(first_key_pos=1),
    "bzmpop": CommandSpec(first_key_pos=1),
    "zunion": CommandSpec(first_key_pos=1),
    "zinter": CommandSpec(first_key_pos=1),
    "zdiff": CommandSpec(first_key_pos=1),
    "zunionstore": CommandSpec(first_key_pos=1),
    "zinterstore": CommandSpec(first_key_pos=1),
    "zdiffstore": CommandSpec(first_key_pos=1),
    "zscan": CommandSpec(first_key_pos=1, key_count=1),
    "zrangestore": CommandSpec(first_key_pos=1, key_count=1),
    # --- streams ------------------------------------------------------------
    "xadd": CommandSpec(first_key_pos=1, key_count=1),
    "xdel": CommandSpec(first_key_pos=1, key_count=1),
    "xlen": CommandSpec(first_key_pos=1, key_count=1),
    "xrange": CommandSpec(first_key_pos=1, key_count=1),
    "xrevrange": CommandSpec(first_key_pos=1, key_count=1),
    "xread": CommandSpec(first_key_pos=1),
    "xreadgroup": CommandSpec(first_key_pos=1),
    "xgroup": CommandSpec(first_key_pos=1, key_count=1),
    "xack": CommandSpec(first_key_pos=1, key_count=1),
    "xclaim": CommandSpec(first_key_pos=1, key_count=1),
    "xautoclaim": CommandSpec(first_key_pos=1, key_count=1),
    "xpending": CommandSpec(first_key_pos=1, key_count=1),
    "xtrim": CommandSpec(first_key_pos=1, key_count=1),
    "xsetid": CommandSpec(first_key_pos=1, key_count=1),
    "xinfo": CommandSpec(first_key_pos=2),
    # --- geospatial --------------------------------------------------------
    "geoadd": CommandSpec(first_key_pos=1, key_count=1),
    "geopos": CommandSpec(first_key_pos=1, key_count=1),
    "geodist": CommandSpec(first_key_pos=1, key_count=1),
    "georadius": CommandSpec(first_key_pos=1),
    "georadiusbymember": CommandSpec(first_key_pos=1),
    "geosearch": CommandSpec(first_key_pos=1, key_count=1),
    "geosearchstore": CommandSpec(first_key_pos=1, key_count=1),
    "geohash": CommandSpec(first_key_pos=1, key_count=1),
    # --- hyperloglog -------------------------------------------------------
    "pfadd": CommandSpec(first_key_pos=1, key_count=1),
    "pfcount": CommandSpec(first_key_pos=1),
    "pfmerge": CommandSpec(first_key_pos=1, key_count=1),
    # --- bitmap ------------------------------------------------------------
    "setbit": CommandSpec(first_key_pos=1, key_count=1),
    "getbit": CommandSpec(first_key_pos=1, key_count=1),
    "bitcount": CommandSpec(first_key_pos=1, key_count=1),
    "bitpos": CommandSpec(first_key_pos=1, key_count=1),
    "bitop": CommandSpec(first_key_pos=1, key_count=1),
    "bitfield": CommandSpec(first_key_pos=1, key_count=1),
    "bitfield_ro": CommandSpec(first_key_pos=1, key_count=1),
    # --- transactions / server (no keys) ----------------------------------
    "multi": CommandSpec(first_key_pos=0),
    "exec": CommandSpec(first_key_pos=0),
    "discard": CommandSpec(first_key_pos=0),
    "watch": CommandSpec(first_key_pos=1),
    "unwatch": CommandSpec(first_key_pos=0),
    "auth": CommandSpec(first_key_pos=0, sensitive_params=(_AUTH_PASSWORD,)),
    "select": CommandSpec(first_key_pos=0),
    "swapdb": CommandSpec(first_key_pos=0),
    "flushdb": CommandSpec(first_key_pos=0),
    "flushall": CommandSpec(first_key_pos=0),
    "info": CommandSpec(first_key_pos=0),
    "ping": CommandSpec(first_key_pos=0),
    "echo": CommandSpec(first_key_pos=0),
    "time": CommandSpec(first_key_pos=0),
    "dbsize": CommandSpec(first_key_pos=0),
    "config": CommandSpec(first_key_pos=0, sensitive_params=(_CONFIG_SET_PASSWORD,)),
    "command": CommandSpec(first_key_pos=0),
    "slowlog": CommandSpec(first_key_pos=0),
    "client": CommandSpec(first_key_pos=0),
    "acl": CommandSpec(first_key_pos=0),
    "module": CommandSpec(first_key_pos=0),
    "monitor": CommandSpec(first_key_pos=0),
    "debug": CommandSpec(first_key_pos=0),
    "shutdown": CommandSpec(first_key_pos=0),
    "save": CommandSpec(first_key_pos=0),
    "bgsave": CommandSpec(first_key_pos=0),
    "bgrewriteaof": CommandSpec(first_key_pos=0),
    "lastsave": CommandSpec(first_key_pos=0),
    "role": CommandSpec(first_key_pos=0),
    "replicaof": CommandSpec(first_key_pos=0),
    "slaveof": CommandSpec(first_key_pos=0),
    "failover": CommandSpec(first_key_pos=0),
    "psync": CommandSpec(first_key_pos=0),
    "sync": CommandSpec(first_key_pos=0),
    "asking": CommandSpec(first_key_pos=0),
    "readonly": CommandSpec(first_key_pos=0),
    "readwrite": CommandSpec(first_key_pos=0),
    "reset": CommandSpec(first_key_pos=0),
    "hello": CommandSpec(first_key_pos=0, sensitive_params=(_AUTH_PASSWORD,)),
    # --- scripting ---------------------------------------------------------
    "eval": CommandSpec(first_key_pos=1),
    "evalsha": CommandSpec(first_key_pos=1),
    "eval_ro": CommandSpec(first_key_pos=1),
    "evalsha_ro": CommandSpec(first_key_pos=1),
    "script": CommandSpec(first_key_pos=0),
    "fcall": CommandSpec(first_key_pos=1),
    "fcall_ro": CommandSpec(first_key_pos=1),
    "function": CommandSpec(first_key_pos=0),
    # --- pubsub ------------------------------------------------------------
    "publish": CommandSpec(first_key_pos=1),
    "subscribe": CommandSpec(first_key_pos=1),
    "unsubscribe": CommandSpec(first_key_pos=1),
    "psubscribe": CommandSpec(first_key_pos=1),
    "punsubscribe": CommandSpec(first_key_pos=1),
    "ssubscribe": CommandSpec(first_key_pos=1),
    "sunsubscribe": CommandSpec(first_key_pos=1),
    "spublish": CommandSpec(first_key_pos=1),
    "pubsub": CommandSpec(first_key_pos=1),
    # --- misc --------------------------------------------------------------
    "migrate": CommandSpec(first_key_pos=0),
    "georadius_ro": CommandSpec(first_key_pos=1),
    "georadiusbymember_ro": CommandSpec(first_key_pos=1),
}


def get_command_spec(command_name: str) -> Optional[CommandSpec]:
    """
    Look up the registered spec for a command.

    The lookup is case-insensitive. Returns ``None`` for commands that are not
    registered, which the inspector reports as *unknown*.
    """

    if not command_name:
        return None
    return COMMAND_SPECS.get(command_name.lower())
