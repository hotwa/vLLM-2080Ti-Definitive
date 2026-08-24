# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


def _tool_calls(parser, chunks):
    calls = []
    for chunk in chunks:
        calls.extend(parser.parse_single_streaming_chunks(chunk).tool_calls)
    return calls


def test_close_tags_split_across_deltas_do_not_duplicate_json_closures():
    parser = StreamingXMLToolCallParser()
    calls = _tool_calls(
        parser,
        [
            "<tool_call><function=search_weather><parameter=city>Shanghai</parameter>",
            "</fun",
            "ction>",
            "</tool",
            "_call>",
        ],
    )

    arguments = "".join(
        call.function.arguments or ""
        for call in calls
        if call.function is not None
    )
    assert arguments.count("{") == 1
    assert arguments.count("}") == 1
    assert arguments.endswith('"}')


def test_multiple_tool_calls_keep_close_tags_scoped_per_call():
    parser = StreamingXMLToolCallParser()
    calls = _tool_calls(
        parser,
        [
            (
                "<tool_call><function=first><parameter=value>one</parameter>"
                "</function></tool_call><tool_call><function=second>"
                "<parameter=value>two</parameter></function></tool_call>"
            )
        ],
    )

    arguments = [
        call.function.arguments
        for call in calls
        if call.function is not None and call.function.arguments
    ]
    assert sum(argument.count("}") for argument in arguments) == 2


def test_close_tag_recovery_uses_active_call_region():
    parser = StreamingXMLToolCallParser()
    calls = _tool_calls(
        parser,
        [
            "<tool_call><function=first><parameter=value>one</parameter>"
            "</function></tool_call><tool_call><function=second>"
            "<parameter=value>two</parameter></function>",
            "</tool_call>",
        ],
    )

    arguments = [
        call.function.arguments
        for call in calls
        if call.function is not None and call.function.arguments
    ]
    assert len(arguments) == 2
    assert all(argument.count("{") == 1 for argument in arguments)
    assert all(argument.count("}") == 1 for argument in arguments)
