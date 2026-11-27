"""One weather tool round for chat-template renders: the tool's schema, the call and the conversation a user turn, that
call, its reply and the answer make."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City."}},
                "required": ["city"],
            },
        },
    }
]
CALL = {"type": "function", "id": "call_0", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
TOOL_ROUND = [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "SUNNY"},
    {"role": "assistant", "content": "It is sunny."},
]
