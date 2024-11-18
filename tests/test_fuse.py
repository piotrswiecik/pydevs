import logging
import datetime


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def test_create_trace(fuse_client):
    trace = fuse_client.create_trace(
        f"trace_id-{timestamp()}", "trace_name_1", "session_id_1", "user_id_123"
    )
    fuse_client.finalize_trace(trace, "input_x", "output_x")
    assert trace.id == "trace_id"
    assert "trace_id" in trace.get_trace_url()


def test_create_span(fuse_client):
    trace = fuse_client.create_trace(
        f"trace_id-{timestamp()}", "trace_name_x", "session_id_2", "user_id_123"
    )
    span = fuse_client.create_span(trace, f"span_name_{timestamp()}", "input_x")
    fuse_client.finalize_span(span, "span_name_x", "input_x", "output_x")
    assert span.id is not None


def test_create_generation(fuse_client):
    trace = fuse_client.create_trace(
        f"trace_id-{timestamp()}", "trace_name_3", "session_id_3", "user_id_123"
    )
    gen = fuse_client.create_generation(trace, f"gen_name_{timestamp()}", "input_x")
    fuse_client.finalize_trace(trace, "input_x", "output_x")
    assert "trace_id" in gen.trace_id
