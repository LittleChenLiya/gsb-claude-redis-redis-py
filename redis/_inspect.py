"""
Offline inspection of staged pipeline commands.

``Pipeline.inspect_commands()`` builds an explicit manifest of everything the
pipeline has staged so far: each command keeps its original sequence number,
the positions of its key arguments, and whether it will run inside the
MULTI/EXEC block. Building the manifest is a purely local operation - it never
acquires a connection, never packs or encodes a command, and never mutates
``command_stack``, so inspecting a pipeline can never change what a later
``execute()`` would send.

Key positions come from the static command registry below (``_KEY_SPECS``),
not from heuristics: a command the registry does not know is reported with
``key_status == "unknown"`` and no key positions, and string arguments are
never guessed to be keys. Sensitive arguments (credentials such as AUTH
passwords) are redacted before anything is stored: the manifest records only
their *category* (``"credential"``) and replaces the value with a placeholder,
so a password never appears in the manifest, its ``repr``, ``to_dict()`` or
``render()`` output.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from redis.utils import str_if_bytes

__all__ = [
    "CREDENTIAL_CATEGORY",
    "InspectedCommand",
    "PipelineInspection",
    "REDACTED_PLACEHOLDER",
    "inspect_command_stack",
]

#: Category reported for arguments that carry a credential (passwords, auth
#: tokens). The value itself is never stored in the manifest.
CREDENTIAL_CATEGORY = "credential"

#: Placeholder substituted for sensitive argument values in the manifest.
REDACTED_PLACEHOLDER = "<credential>"


@dataclass(frozen=True)
class _KeyRange:
    """
    Positional key spec in command-relative coordinates.

    ``first``/``last`` are 1-based argument positions where position 1 is the
    first argument *after* the command name; a negative ``last`` counts back
    from the final argument (``-1`` is the last argument, ``-2`` the one
    before it, ...). ``step`` strides the range (``MSET`` uses step 2).
    """

    first: int
    last: int = -1
    step: int = 1

    def positions(self, nargs: int, offset: int) -> Tuple[int, ...]:
        """Absolute argument indexes for a command of ``nargs`` tokens whose
        name occupies ``offset`` leading tokens."""
        start = offset + self.first - 1
        end = offset + self.last - 1 if self.last > 0 else nargs + self.last
        if end < start:
            return ()
        return tuple(range(start, end + 1, self.step))


_ONE = _KeyRange(1, 1)
_TWO = _KeyRange(1, 2)
_ALL = _KeyRange(1, -1)

# Spec value types: a _KeyRange, a callable (args, offset) -> Tuple[int, ...],
# or None for a known command that takes no keys.
_KeySpec = Union[_KeyRange, Callable[[Sequence[Any], int], Tuple[int, ...]], None]


def _numkeys_from(position: int, keys_from: int) -> Callable:
    """Keys are ``args[keys_from : keys_from + int(args[position])]``
    (EVAL/FCALL style, positions relative to the command offset)."""

    def handler(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
        try:
            numkeys = int(args[offset + position])
        except (IndexError, TypeError, ValueError):
            return ()
        start = offset + keys_from
        return tuple(range(start, min(start + max(numkeys, 0), len(args))))

    return handler


def _streams_keys(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
    """XREAD/XREADGROUP: keys are the first half of the tokens after STREAMS."""
    for i in range(offset, len(args)):
        if str_if_bytes(args[i]).upper() == "STREAMS":
            rest = len(args) - i - 1
            count = rest // 2
            return tuple(range(i + 1, i + 1 + count))
    return ()


def _numkeys_command(position: int) -> Callable:
    """ZUNION/ZINTER/ZINTERCARD/LMPOP style: ``numkeys`` then that many keys."""

    def handler(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
        try:
            numkeys = int(args[offset + position])
        except (IndexError, TypeError, ValueError):
            return ()
        start = offset + position + 1
        return tuple(range(start, min(start + max(numkeys, 0), len(args))))

    return handler


def _store_numkeys_command(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
    """ZUNIONSTORE/ZINTERSTORE/ZDIFFSTORE: destination, numkeys, keys."""
    try:
        numkeys = int(args[offset + 1])
    except (IndexError, TypeError, ValueError):
        return (offset,) if len(args) > offset else ()
    start = offset + 2
    return (offset,) + tuple(range(start, min(start + max(numkeys, 0), len(args))))


def _timeout_numkeys_command(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
    """BLMPOP/BZMPOP: timeout, numkeys, keys."""

    try:
        numkeys = int(args[offset + 1])
    except (IndexError, TypeError, ValueError):
        return ()
    start = offset + 2
    return tuple(range(start, min(start + max(numkeys, 0), len(args))))


def _bitop_keys(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
    """BITOP: NOT takes dest+src; the other operations take dest+src...."""
    if len(args) <= offset + 2:
        return ()
    if str_if_bytes(args[offset]).upper() == "NOT":
        return (offset + 1, offset + 2)
    return tuple(range(offset + 1, len(args)))


def _sort_keys(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
    """SORT: the key, plus the STORE destination when present."""
    positions = [offset] if len(args) > offset else []
    for i in range(offset + 1, len(args) - 1):
        if str_if_bytes(args[i]).upper() == "STORE":
            positions.append(i + 1)
            break
    return tuple(positions)


def _georadius_keys(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
    """GEORADIUS/GEORADIUSBYMEMBER: the key, plus STORE/STOREDIST targets."""
    positions = [offset] if len(args) > offset else []
    for i in range(offset + 1, len(args) - 1):
        if str_if_bytes(args[i]).upper() in ("STORE", "STOREDIST"):
            positions.append(i + 1)
    return tuple(positions)


def _migrate_keys(args: Sequence[Any], offset: int) -> Tuple[int, ...]:
    """MIGRATE: the single key after host/port, or the KEYS list at the end."""
    for i in range(offset, len(args)):
        if str_if_bytes(args[i]).upper() == "KEYS":
            return tuple(range(i + 1, len(args)))
    if len(args) > offset + 2 and args[offset + 2] not in ("", b""):
        return (offset + 2,)
    return ()


def _register(spec: _KeySpec, *names: str) -> None:
    for name in names:
        _KEY_SPECS[name] = spec


_KEY_SPECS: Dict[str, _KeySpec] = {}

# Strings
_register(
    _ONE,
    "GET",
    "GETDEL",
    "GETEX",
    "GETRANGE",
    "GETSET",
    "SET",
    "SETEX",
    "SETNX",
    "SETRANGE",
    "PSETEX",
    "STRLEN",
    "APPEND",
    "SUBSTR",
    "INCR",
    "DECR",
    "INCRBY",
    "INCRBYFLOAT",
    "DECRBY",
    "GETBIT",
    "SETBIT",
    "BITCOUNT",
    "BITPOS",
    "BITFIELD",
    "BITFIELD_RO",
)
_register(_ALL, "MGET")
_register(_KeyRange(1, -1, 2), "MSET", "MSETNX")
_register(_bitop_keys, "BITOP")

# Generic key commands
_register(
    _ONE,
    "TYPE",
    "DUMP",
    "MOVE",
    "TTL",
    "PTTL",
    "EXPIRE",
    "PEXPIRE",
    "EXPIREAT",
    "PEXPIREAT",
    "EXPIRETIME",
    "PEXPIRETIME",
    "PERSIST",
    "RESTORE",
)
_register(_TWO, "RENAME", "RENAMENX", "COPY")
_register(_ALL, "DEL", "UNLINK", "EXISTS", "TOUCH", "WATCH")
_register(_sort_keys, "SORT")
_register(_ONE, "SORT_RO")
_register(_migrate_keys, "MIGRATE")

# Lists
_register(
    _ONE,
    "LPUSH",
    "RPUSH",
    "LPUSHX",
    "RPUSHX",
    "LPOP",
    "RPOP",
    "LLEN",
    "LRANGE",
    "LINDEX",
    "LSET",
    "LREM",
    "LTRIM",
    "LINSERT",
    "LPOS",
)
_register(_TWO, "RPOPLPUSH", "BRPOPLPUSH", "LMOVE", "BLMOVE")
_register(_KeyRange(1, -2), "BLPOP", "BRPOP")
_register(_numkeys_command(0), "LMPOP")
_register(_timeout_numkeys_command, "BLMPOP")

# Sets
_register(
    _ONE,
    "SADD",
    "SREM",
    "SMEMBERS",
    "SCARD",
    "SISMEMBER",
    "SMISMEMBER",
    "SPOP",
    "SRANDMEMBER",
    "SSCAN",
)
_register(_TWO, "SMOVE")
_register(
    _ALL,
    "SDIFF",
    "SINTER",
    "SUNION",
    "SDIFFSTORE",
    "SINTERSTORE",
    "SUNIONSTORE",
)
_register(_numkeys_command(0), "SINTERCARD")

# Hashes
_register(
    _ONE,
    "HSET",
    "HGET",
    "HDEL",
    "HEXISTS",
    "HGETALL",
    "HKEYS",
    "HVALS",
    "HLEN",
    "HMGET",
    "HMSET",
    "HINCRBY",
    "HINCRBYFLOAT",
    "HRANDFIELD",
    "HSCAN",
    "HSETNX",
    "HSTRLEN",
)

# Sorted sets
_register(
    _ONE,
    "ZADD",
    "ZREM",
    "ZSCORE",
    "ZMSCORE",
    "ZCARD",
    "ZCOUNT",
    "ZLEXCOUNT",
    "ZINCRBY",
    "ZRANGE",
    "ZREVRANGE",
    "ZRANGEBYSCORE",
    "ZREVRANGEBYSCORE",
    "ZRANGEBYLEX",
    "ZREVRANGEBYLEX",
    "ZRANK",
    "ZREVRANK",
    "ZPOPMIN",
    "ZPOPMAX",
    "ZSCAN",
    "ZRANDMEMBER",
)
_register(_TWO, "ZRANGESTORE")
_register(_KeyRange(1, -2), "BZPOPMIN", "BZPOPMAX")
_register(_numkeys_command(0), "ZMPOP", "ZUNION", "ZINTER", "ZDIFF", "ZINTERCARD")
_register(_timeout_numkeys_command, "BZMPOP")
_register(_store_numkeys_command, "ZUNIONSTORE", "ZINTERSTORE", "ZDIFFSTORE")

# HyperLogLog
_register(_ONE, "PFADD")
_register(_ALL, "PFCOUNT", "PFMERGE")

# Geospatial
_register(_ONE, "GEOADD", "GEODIST", "GEOHASH", "GEOPOS", "GEOSEARCH")
_register(_TWO, "GEOSEARCHSTORE")
_register(
    _georadius_keys,
    "GEORADIUS",
    "GEORADIUSBYMEMBER",
    "GEORADIUS_RO",
    "GEORADIUSBYMEMBER_RO",
)

# Streams
_register(
    _ONE,
    "XADD",
    "XLEN",
    "XRANGE",
    "XREVRANGE",
    "XDEL",
    "XTRIM",
    "XACK",
    "XPENDING",
    "XCLAIM",
    "XAUTOCLAIM",
    "XSETID",
)
_register(_streams_keys, "XREAD", "XREADGROUP")
_register(_ONE, "XINFO STREAM", "XINFO GROUPS", "XINFO CONSUMERS")
_register(
    _ONE,
    "XGROUP CREATE",
    "XGROUP SETID",
    "XGROUP DESTROY",
    "XGROUP DELCONSUMER",
    "XGROUP CREATECONSUMER",
)

# Scripting and functions
_register(
    _numkeys_from(1, 2),
    "EVAL",
    "EVALSHA",
    "EVAL_RO",
    "EVALSHA_RO",
    "FCALL",
    "FCALL_RO",
)

# Introspection subcommands that take a key
_register(
    _ONE,
    "OBJECT ENCODING",
    "OBJECT FREQ",
    "OBJECT IDLETIME",
    "OBJECT REFCOUNT",
    "MEMORY USAGE",
)

# Known commands without key arguments (channels and patterns are not keys).
_NO_KEY_COMMANDS = (
    "PING",
    "ECHO",
    "AUTH",
    "HELLO",
    "SELECT",
    "SWAPDB",
    "QUIT",
    "RESET",
    "CLIENT",
    "COMMAND",
    "CONFIG",
    "CONFIG GET",
    "CONFIG SET",
    "CONFIG RESETSTAT",
    "CONFIG REWRITE",
    "DBSIZE",
    "DEBUG",
    "FAILOVER",
    "FLUSHALL",
    "FLUSHDB",
    "INFO",
    "KEYS",
    "LASTSAVE",
    "LATENCY",
    "LOLWUT",
    "MEMORY",
    "MEMORY DOCTOR",
    "MEMORY STATS",
    "MONITOR",
    "MODULE",
    "PSYNC",
    "PUBSUB",
    "RANDOMKEY",
    "REPLICAOF",
    "ROLE",
    "SAVE",
    "BGSAVE",
    "BGREWRITEAOF",
    "SCAN",
    "SCRIPT",
    "SCRIPT LOAD",
    "SCRIPT EXISTS",
    "SCRIPT KILL",
    "SCRIPT FLUSH",
    "SHUTDOWN",
    "SLAVEOF",
    "SLOWLOG",
    "SYNC",
    "TIME",
    "WAIT",
    "ACL",
    "ACL SETUSER",
    "ACL GETUSER",
    "ACL DELUSER",
    "ACL LIST",
    "ACL WHOAMI",
    "CLUSTER",
    "FUNCTION",
    "MULTI",
    "EXEC",
    "DISCARD",
    "UNWATCH",
    "SUBSCRIBE",
    "UNSUBSCRIBE",
    "PSUBSCRIBE",
    "PUNSUBSCRIBE",
    "SSUBSCRIBE",
    "SUNSUBSCRIBE",
    "PUBLISH",
    "SPUBLISH",
)
_register(None, *_NO_KEY_COMMANDS)


def _auth_sensitive(args: Sequence[Any], offset: int) -> Tuple[Tuple[int, str], ...]:
    """AUTH [username] password - the password is always the last argument."""
    if len(args) - offset >= 1:
        return ((len(args) - 1, CREDENTIAL_CATEGORY),)
    return ()


def _hello_sensitive(args: Sequence[Any], offset: int) -> Tuple[Tuple[int, str], ...]:
    """HELLO [protover [AUTH user password] ...] - password follows AUTH user."""
    for i in range(offset, len(args) - 2):
        if str_if_bytes(args[i]).upper() == "AUTH":
            return ((i + 2, CREDENTIAL_CATEGORY),)
    return ()


def _migrate_sensitive(args: Sequence[Any], offset: int) -> Tuple[Tuple[int, str], ...]:
    """MIGRATE ... AUTH password / AUTH2 user password."""
    found = []
    for i in range(offset, len(args)):
        token = str_if_bytes(args[i]).upper()
        if token == "AUTH" and i + 1 < len(args):
            found.append((i + 1, CREDENTIAL_CATEGORY))
        elif token == "AUTH2" and i + 2 < len(args):
            found.append((i + 2, CREDENTIAL_CATEGORY))
    return tuple(found)


def _acl_setuser_sensitive(
    args: Sequence[Any], offset: int
) -> Tuple[Tuple[int, str], ...]:
    """ACL SETUSER rules: ``>password`` and ``#hashedpassword`` carry secrets."""
    found = []
    for i in range(offset + 1, len(args)):
        rule = str_if_bytes(args[i])
        if rule.startswith((">", "#")):
            found.append((i, CREDENTIAL_CATEGORY))
    return tuple(found)


_SENSITIVE_CONFIG_KEYS = frozenset({"requirepass", "masterauth"})


def _config_set_sensitive(
    args: Sequence[Any], offset: int
) -> Tuple[Tuple[int, str], ...]:
    """CONFIG SET name value [name value ...] - credential-carrying options."""
    found = []
    for i in range(offset, len(args) - 1, 2):
        if str_if_bytes(args[i]).lower() in _SENSITIVE_CONFIG_KEYS:
            found.append((i + 1, CREDENTIAL_CATEGORY))
    return tuple(found)


_SENSITIVE_SPECS: Dict[
    str, Callable[[Sequence[Any], int], Tuple[Tuple[int, str], ...]]
] = {
    "AUTH": _auth_sensitive,
    "HELLO": _hello_sensitive,
    "MIGRATE": _migrate_sensitive,
    "ACL SETUSER": _acl_setuser_sensitive,
    "CONFIG SET": _config_set_sensitive,
}


def _resolve_command(args: Sequence[Any]) -> Tuple[str, int, bool]:
    """
    Resolve the canonical (uppercased) command name and its token offset.

    Handles both spellings of container subcommands - ``("CONFIG", "SET", ...)``
    and ``("CONFIG SET", ...)`` - and reports whether the command is in the
    registry at all. The two-token form wins so a container command such as
    ``("CONFIG", "SET", ...)`` is not mistaken for the bare ``CONFIG`` entry.
    """
    first = str_if_bytes(args[0]).upper()
    if len(args) > 1:
        two = f"{first} {str_if_bytes(args[1]).upper()}"
        if two in _KEY_SPECS or two in _SENSITIVE_SPECS:
            return two, 2, True
    if first in _KEY_SPECS or first in _SENSITIVE_SPECS:
        return first, 1, True
    return first, 1, False


def _key_positions(
    name: str, args: Sequence[Any], offset: int
) -> Optional[Tuple[int, ...]]:
    spec = _KEY_SPECS.get(name)
    if spec is None:
        return ()
    if isinstance(spec, _KeyRange):
        return spec.positions(len(args), offset)
    return tuple(spec(args, offset))


@dataclass(frozen=True)
class InspectedCommand:
    """
    One staged command as recorded by the offline manifest.

    ``args`` is the staged argument tuple with sensitive values replaced by
    :data:`REDACTED_PLACEHOLDER`; the original values are never stored.
    ``key_positions`` holds absolute indexes into ``args`` (index 0 is the
    command name) and is ``None`` when the command is not in the registry -
    check :attr:`key_status` rather than guessing.
    """

    index: int
    command: str
    args: Tuple[Any, ...]
    keys: Tuple[Any, ...]
    key_positions: Optional[Tuple[int, ...]]
    in_transaction: bool
    sensitive: Tuple[str, ...] = ()

    @property
    def key_status(self) -> str:
        """``"known"`` when key positions follow the registered command spec
        (possibly empty for keyless commands), ``"unknown"`` otherwise."""
        return "unknown" if self.key_positions is None else "known"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "command": self.command,
            "args": self.args,
            "keys": self.keys,
            "key_positions": self.key_positions,
            "key_status": self.key_status,
            "in_transaction": self.in_transaction,
            "sensitive": self.sensitive,
        }

    def render(self) -> str:
        parts = [f"#{self.index} {self.command}"]
        if self.key_positions is None:
            parts.append("key_status=unknown")
        else:
            parts.append(f"keys={self.keys!r}")
        if self.sensitive:
            parts.append(f"sensitive={self.sensitive!r}")
        return " ".join(parts)


@dataclass(frozen=True)
class PipelineInspection:
    """
    Offline manifest of a pipeline's staged commands, in staging order.

    ``transaction`` is True when the staged commands will be wrapped in
    MULTI/EXEC (transactional pipeline or explicit ``multi()``); in that case
    :meth:`render` shows the MULTI/EXEC boundaries around the command list so
    the transaction order is visible.
    """

    transaction: bool
    commands: Tuple[InspectedCommand, ...]

    def __len__(self) -> int:
        return len(self.commands)

    def __iter__(self):
        return iter(self.commands)

    def __getitem__(self, item):
        return self.commands[item]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction": self.transaction,
            "commands": [command.to_dict() for command in self.commands],
        }

    def render(self) -> str:
        """Text protocol of the manifest, one line per staged command."""
        lines = [
            f"PIPELINE transaction={self.transaction} commands={len(self.commands)}"
        ]
        if self.transaction:
            lines.append("MULTI")
        for command in self.commands:
            line = command.render()
            lines.append(f"  {line}" if self.transaction else line)
        if self.transaction:
            lines.append("EXEC")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def inspect_command_stack(command_stack, transaction: bool) -> PipelineInspection:
    """
    Build a :class:`PipelineInspection` from a pipeline's ``command_stack``.

    Purely local: reads the staged ``(args, options)`` entries without
    encoding, packing, or mutating anything. ``transaction`` is the pipeline's
    effective transaction flag (``transaction or explicit_transaction``); every
    staged command sits inside the MULTI/EXEC block when it is set.
    """
    inspected = []
    for index, entry in enumerate(command_stack):
        args = tuple(entry[0])
        name, offset, known = _resolve_command(args)

        sensitive_indexes: Dict[int, str] = {}
        sensitive_handler = _SENSITIVE_SPECS.get(name)
        if sensitive_handler is not None:
            for position, category in sensitive_handler(args, offset):
                sensitive_indexes[position] = category

        if sensitive_indexes:
            redacted = tuple(
                REDACTED_PLACEHOLDER if i in sensitive_indexes else value
                for i, value in enumerate(args)
            )
        else:
            redacted = args

        if known:
            positions = _key_positions(name, args, offset)
            keys = tuple(args[position] for position in positions)
        else:
            positions = None
            keys = ()

        inspected.append(
            InspectedCommand(
                index=index,
                command=name,
                args=redacted,
                keys=keys,
                key_positions=positions,
                in_transaction=transaction,
                sensitive=tuple(sorted(set(sensitive_indexes.values()))),
            )
        )
    return PipelineInspection(transaction=transaction, commands=tuple(inspected))
