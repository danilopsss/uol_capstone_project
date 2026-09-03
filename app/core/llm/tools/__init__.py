from ollama import Tool

_SIMPLIFY_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "simplify_legal_doc",
        "description": "This is called eveytime a summarization is required.",
        "parameters": {
            "type": "object",
            "properties": {
                "document": {
                    "type": "string",
                    "description": "The legal document text to simplify.",
                }
            },
            "required": ["document"],
        },
    },
}

TOOLS: list[Tool] = [Tool.model_validate(_SIMPLIFY_TOOL_SCHEMA)]
