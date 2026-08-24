# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


def _deltas(parser, chunks):
    return [parser.parse_single_streaming_chunks(chunk) for chunk in chunks]


def _tool_calls(deltas):
    return [tool_call for delta in deltas for tool_call in delta.tool_calls]


def test_function_name_is_only_emitted_in_first_delta():
    parser = StreamingXMLToolCallParser()
    deltas = _deltas(
        parser,
        [
            "<tool_call>",
            "<function=search_weather>",
            "<parameter=city>Shanghai</parameter>",
            "</function>",
            "</tool_call>",
        ],
    )

    calls = _tool_calls(deltas)
    assert calls
    assert calls[0].function is not None
    assert calls[0].function.name == "search_weather"
    assert all(
        call.function is None or call.function.name is None for call in calls[1:]
    )
    assert all(
        "name" not in call.function.model_dump(exclude_unset=True)
        for call in calls[1:]
        if call.function is not None
    )


def test_function_name_is_emitted_once_for_each_tool_call():
    parser = StreamingXMLToolCallParser()
    deltas = _deltas(
        parser,
        [
            "<tool_call>",
            "<function=search_weather>",
            "<parameter=city>Shanghai</parameter>",
            "</function>",
            "</tool_call>",
            "<tool_call>",
            "<function=get_time>",
            "<parameter=timezone>UTC</parameter>",
            "</function>",
            "</tool_call>",
        ],
    )

    calls_by_index = {}
    for call in _tool_calls(deltas):
        calls_by_index.setdefault(call.index, []).append(call)

    assert set(calls_by_index) == {0, 1}
    for expected_name, calls in zip(
        ("search_weather", "get_time"),
        (calls_by_index[0], calls_by_index[1]),
        strict=True,
    ):
        assert calls[0].function is not None
        assert calls[0].function.name == expected_name
        assert all(
            call.function is None or call.function.name is None for call in calls[1:]
        )
        assert all(
            "name" not in call.function.model_dump(exclude_unset=True)
            for call in calls[1:]
            if call.function is not None
        )
