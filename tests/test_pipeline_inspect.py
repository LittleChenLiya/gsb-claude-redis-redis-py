from redis._inspect import (
    CREDENTIAL_CATEGORY,
    REDACTED_PLACEHOLDER,
    PipelineInspection,
)
from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.client import Pipeline


class FakeConnection:
    """A connection that fails the test if any I/O or encoding is attempted."""

    def __init__(self):
        self.encoder = self

    def encode(self, *args, **kwargs):
        raise AssertionError("inspect must not encode anything")

    def pack_commands(self, *args, **kwargs):
        raise AssertionError("inspect must not pack commands")

    def send_command(self, *args, **kwargs):
        raise AssertionError("inspect must not send commands")

    def send_packed_command(self, *args, **kwargs):
        raise AssertionError("inspect must not send commands")


class FakeConnectionPool:
    """A pool that fails the test if a connection is ever requested."""

    def __init__(self):
        self.connection = FakeConnection()

    def get_connection(self, *args, **kwargs):
        raise AssertionError("inspect must not acquire a connection")

    def release(self, *args, **kwargs):
        raise AssertionError("inspect must not touch the pool")


def make_pipeline(transaction=True):
    return Pipeline(FakeConnectionPool(), {}, transaction, None)


def make_async_pipeline(transaction=True):
    return AsyncPipeline(FakeConnectionPool(), {}, transaction, None)


class TestInspectKnownKeys:
    def test_get_reports_known_key(self):
        pipe = make_pipeline()
        pipe.get("foo")
        manifest = pipe.inspect_commands()
        assert isinstance(manifest, PipelineInspection)
        assert len(manifest) == 1
        command = manifest[0]
        assert command.index == 0
        assert command.command == "GET"
        assert command.key_status == "known"
        assert command.keys == ("foo",)
        assert command.key_positions == (1,)
        assert command.in_transaction is True
        assert command.sensitive == ()

    def test_mset_reports_every_key_position(self):
        pipe = make_pipeline()
        pipe.mset({"a": 1, "b": 2, "c": 3})
        (command,) = pipe.inspect_commands()
        assert command.command == "MSET"
        assert command.key_status == "known"
        assert command.keys == ("a", "b", "c")
        assert command.key_positions == (1, 3, 5)

    def test_key_positions_follow_registered_specs(self):
        pipe = make_pipeline()
        pipe.execute_command("EVAL", "return 1", 2, "k1", "k2", "arg")
        pipe.execute_command("BLPOP", "l1", "l2", 5)
        pipe.execute_command("ZUNIONSTORE", "dest", 2, "z1", "z2")
        eval_cmd, blpop_cmd, zstore_cmd = pipe.inspect_commands()
        assert eval_cmd.keys == ("k1", "k2")
        assert eval_cmd.key_positions == (3, 4)
        # the trailing timeout is not a key
        assert blpop_cmd.keys == ("l1", "l2")
        assert blpop_cmd.key_positions == (1, 2)
        assert zstore_cmd.keys == ("dest", "z1", "z2")
        assert zstore_cmd.key_positions == (1, 3, 4)

    def test_keyless_command_is_known_with_no_keys(self):
        pipe = make_pipeline()
        pipe.ping()
        (command,) = pipe.inspect_commands()
        assert command.command == "PING"
        assert command.key_status == "known"
        assert command.keys == ()
        assert command.key_positions == ()


class TestInspectUnknownCommands:
    def test_custom_command_is_unknown(self):
        pipe = make_pipeline()
        pipe.execute_command("MYCUSTOM", "k1", "v1")
        (command,) = pipe.inspect_commands()
        assert command.command == "MYCUSTOM"
        assert command.key_status == "unknown"
        assert command.key_positions is None
        assert command.keys == ()

    def test_string_arguments_are_not_guessed_as_keys(self):
        pipe = make_pipeline()
        pipe.execute_command("MODULE.LISTEX", "some-string", "another")
        (command,) = pipe.inspect_commands()
        # not every string argument may be reported as a key
        assert command.keys == ()
        assert "some-string" not in command.keys
        assert command.key_status == "unknown"


class TestInspectSensitiveArguments:
    def test_auth_password_never_appears(self):
        pipe = make_pipeline()
        pipe.execute_command("AUTH", "default", "s3cret-pass")
        manifest = pipe.inspect_commands()
        (command,) = manifest
        assert command.sensitive == (CREDENTIAL_CATEGORY,)
        assert command.args == ("AUTH", "default", REDACTED_PLACEHOLDER)
        for rendered in (
            str(manifest),
            repr(manifest),
            manifest.render(),
            str(manifest.to_dict()),
            repr(command),
        ):
            assert "s3cret-pass" not in rendered

    def test_config_set_requirepass_is_redacted(self):
        pipe = make_pipeline()
        pipe.execute_command("CONFIG", "SET", "requirepass", "t0p-secret")
        pipe.execute_command("CONFIG SET", "maxmemory", "100mb")
        first, second = pipe.inspect_commands()
        assert first.command == "CONFIG SET"
        assert first.sensitive == (CREDENTIAL_CATEGORY,)
        assert "t0p-secret" not in str(pipe.inspect_commands().to_dict())
        # non-credential CONFIG SET options are left alone
        assert second.sensitive == ()
        assert second.args == ("CONFIG SET", "maxmemory", "100mb")

    def test_acl_setuser_password_rule_is_redacted(self):
        pipe = make_pipeline()
        pipe.execute_command("ACL", "SETUSER", "bob", ">hunter2", "on")
        (command,) = pipe.inspect_commands()
        assert command.command == "ACL SETUSER"
        assert command.sensitive == (CREDENTIAL_CATEGORY,)
        assert "hunter2" not in repr(command)


class TestInspectIsOffline:
    def test_command_stack_is_unchanged(self):
        pipe = make_pipeline()
        pipe.set("a", "1")
        pipe.get("a")
        before = list(pipe.command_stack)
        manifest = pipe.inspect_commands()
        assert pipe.command_stack == before
        assert len(pipe.command_stack) == 2
        # a second inspection sees the same stack
        assert pipe.inspect_commands().to_dict() == manifest.to_dict()

    def test_no_connection_is_acquired(self):
        # the fake pool raises if anything asks it for a connection
        pipe = make_pipeline()
        pipe.get("foo")
        pipe.inspect_commands()

    def test_options_are_not_required(self):
        pipe = make_pipeline()
        pipe.execute_command("GET", "k", **{"EMPTY_RESPONSE": "x"})
        (command,) = pipe.inspect_commands()
        assert command.keys == ("k",)


class TestInspectTransactionBoundaries:
    def test_transactional_pipeline_marks_every_command(self):
        pipe = make_pipeline(transaction=True)
        pipe.set("a", "1")
        pipe.get("a")
        manifest = pipe.inspect_commands()
        assert manifest.transaction is True
        assert [c.in_transaction for c in manifest] == [True, True]
        lines = manifest.render().splitlines()
        assert lines[0] == "PIPELINE transaction=True commands=2"
        assert lines[1] == "MULTI"
        assert lines[-1] == "EXEC"
        # the transaction order is shown, in staging order
        assert "  #0 SET" in lines[2]
        assert "  #1 GET" in lines[3]

    def test_non_transactional_pipeline_has_no_multi_exec(self):
        pipe = make_pipeline(transaction=False)
        pipe.set("a", "1")
        pipe.get("a")
        manifest = pipe.inspect_commands()
        assert manifest.transaction is False
        assert [c.in_transaction for c in manifest] == [False, False]
        rendered = manifest.render()
        assert "MULTI" not in rendered
        assert "EXEC" not in rendered
        assert rendered.splitlines()[1] == "#0 SET keys=('a',)"

    def test_explicit_multi_marks_transaction(self):
        pipe = make_pipeline(transaction=False)
        pipe.multi()
        pipe.set("a", "1")
        manifest = pipe.inspect_commands()
        assert manifest.transaction is True
        assert manifest[0].in_transaction is True

    def test_local_commands_are_classified(self):
        pipe = make_pipeline()
        # WATCH is sent immediately by execute_command (it never reaches the
        # stack outside an explicit transaction), so stage it directly.
        pipe.pipeline_execute_command("WATCH", "w1", "w2")
        pipe.execute_command("EXEC")
        pipe.execute_command("DISCARD")
        watch, exec_cmd, discard = pipe.inspect_commands()
        assert watch.key_status == "known"
        assert watch.keys == ("w1", "w2")
        assert exec_cmd.key_status == "known"
        assert exec_cmd.keys == ()
        assert discard.key_status == "known"
        assert discard.keys == ()


class TestInspectOutputProtocol:
    def test_to_dict_structure(self):
        pipe = make_pipeline()
        pipe.get("foo")
        pipe.execute_command("MYCUSTOM", "x")
        manifest = pipe.inspect_commands()
        data = manifest.to_dict()
        assert data["transaction"] is True
        assert [c["index"] for c in data["commands"]] == [0, 1]
        first, second = data["commands"]
        assert first == {
            "index": 0,
            "command": "GET",
            "args": ("GET", "foo"),
            "keys": ("foo",),
            "key_positions": (1,),
            "key_status": "known",
            "in_transaction": True,
            "sensitive": (),
        }
        assert second["key_status"] == "unknown"
        assert second["key_positions"] is None
        assert second["keys"] == ()

    def test_render_is_str(self):
        pipe = make_pipeline()
        pipe.get("foo")
        manifest = pipe.inspect_commands()
        assert str(manifest) == manifest.render()

    def test_empty_pipeline(self):
        pipe = make_pipeline()
        manifest = pipe.inspect_commands()
        assert len(manifest) == 0
        assert manifest.to_dict() == {"transaction": True, "commands": []}
        assert manifest.render() == (
            "PIPELINE transaction=True commands=0\nMULTI\nEXEC"
        )


class TestAsyncPipelineInspect:
    async def test_async_pipeline_inspect(self):
        pipe = make_async_pipeline()
        pipe.get("foo")
        pipe.execute_command("AUTH", "default", "s3cret-pass")
        before = list(pipe.command_stack)
        manifest = pipe.inspect_commands()
        assert pipe.command_stack == before
        get_cmd, auth_cmd = manifest
        assert get_cmd.keys == ("foo",)
        assert get_cmd.in_transaction is True
        assert auth_cmd.sensitive == (CREDENTIAL_CATEGORY,)
        assert "s3cret-pass" not in manifest.render()

    async def test_async_non_transactional(self):
        pipe = make_async_pipeline(transaction=False)
        pipe.mset({"a": 1, "b": 2})
        manifest = pipe.inspect_commands()
        assert manifest.transaction is False
        (command,) = manifest
        assert command.keys == ("a", "b")
        assert command.in_transaction is False
