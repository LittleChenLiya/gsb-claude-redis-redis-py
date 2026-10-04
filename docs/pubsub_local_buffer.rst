.. _pubsub_local_buffer:

Bounded local PubSub buffer with ack/nack
==========================================

``redis.pubsub_buffer.BoundedPubSubBuffer`` is a consumer-side adapter that
sits in front of a PubSub-like source (anything exposing
``get_message(timeout=...)``, e.g. :class:`redis.client.PubSub`) and adds a
bounded local buffer with explicit, **local-only** acknowledgements.

Local semantics -- read this first
-----------------------------------

Everything this adapter does is in-process bookkeeping:

* **A local ack is not a server-side acknowledgement.** Redis Pub/Sub has
  no acknowledgement concept, and this adapter never pretends otherwise:
  ``ack()``/``nack()`` issue no Redis commands and nothing is persisted to
  Redis or anywhere else. If the process dies, buffered and in-flight
  messages are gone.
* **Bounded buffer, explicit backpressure.** At most ``capacity`` messages
  are held locally at any time (buffered plus leased-but-unsettled). When
  the buffer is full the reader thread stops calling the source's
  ``get_message`` until a slot frees up.
* **A leased message is owned by its lease holder.** Until it is acked or
  nacked it is never released from local tracking and never handed to
  another consumer.

Leasing, acking, nacking
-------------------------

.. code-block:: python

    import redis
    from redis.pubsub_buffer import BoundedPubSubBuffer

    client = redis.Redis()
    pubsub = client.pubsub()
    pubsub.subscribe("events")

    def on_failure(message, redeliveries):
        print("gave up on", message, "after", redeliveries, "tries")

    buf = BoundedPubSubBuffer(pubsub, capacity=100,
                              max_redeliveries=3,
                              on_failure=on_failure)

    lease = buf.get(timeout=1.0)     # lease the next buffered message
    if lease is not None:
        try:
            process(lease.message)
        except TransientError:
            lease.nack()             # back to the front of the queue
        else:
            lease.ack()              # release the local slot

* ``lease.ack()`` settles the lease and frees its buffer slot. It returns
  ``True`` for the call that settled the lease; duplicate ``ack()``/``nack()``
  calls are no-ops returning ``False``.
* ``lease.nack()`` puts the message back at the **front** of the local queue
  (so it is redelivered before anything buffered behind it) and increments
  the message's local redelivery counter. When the counter reaches
  ``max_redeliveries``, the message is dropped and ``on_failure(message,
  redeliveries)`` is invoked **exactly once**.

Closing
--------

``close(wait=True)`` stops the reader and blocks (up to an optional
``timeout``) until every in-flight lease has been settled via ack/nack.
``close(wait=False)`` terminates immediately.

Both forms **return a snapshot of the messages still unacknowledged** at
termination -- leased-but-unsettled first, then buffered-but-never-leased
in queue order -- so nothing is silently dropped:

.. code-block:: python

    unacked = buf.close(wait=False)
    for message in unacked:
        requeue_somewhere_else(message)

After termination, late ``ack()``/``nack()`` calls on outstanding leases are
no-ops returning ``False``: the returned snapshot is the authoritative
record of what was left unacknowledged. ``get()`` on a closing/closed buffer
raises :class:`redis.pubsub_buffer.ConsumerClosedError`.
