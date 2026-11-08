import json
import os

from anthropic import Anthropic


MAX_INSPECTIONS = 6
INSPECT = {
    "name": "inspect_columns",
    "description": "Read norms and a sample of values from up to eight consecutive weight columns.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tensor": {"type": "string"},
            "start": {"type": "integer", "minimum": 0},
            "count": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "required": ["tensor", "start", "count"],
        "additionalProperties": False,
    },
}
PROPOSE = {
    "name": "propose_patch",
    "description": "Submit one candidate weight patch for evaluation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tensor": {"type": "string"},
                        "start": {"type": "integer", "minimum": 0},
                        "count": {"type": "integer", "minimum": 1},
                        "operation": {"enum": ["scale", "project_out"]},
                        "strength": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 2,
                            "description": "Scale multiplier, or fraction of the projection to remove (0–1).",
                        },
                        "reference": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Original column used as direction for project_out. Omit for scale.",
                        },
                    },
                    "required": ["tensor", "start", "count", "operation", "strength"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 8,
            },
        },
        "required": ["rationale", "edits"],
        "additionalProperties": False,
    },
}


def client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY with a key from the Claude Console.")
    return Anthropic(api_key=key, timeout=120, max_retries=1)


def propose(api, model, goal, revision, weights):
    context = {
        "goal": goal,
        "checkpoint": revision,
        "tensor_shapes": weights.catalog,
        "parameter_budget": weights.budget,
    }
    messages = [{"role": "user", "content": json.dumps(context)}]
    remaining = MAX_INSPECTIONS
    for _ in range(MAX_INSPECTIONS + 1):
        tools = [INSPECT, PROPOSE] if remaining else [PROPOSE]
        response = api.messages.create(
            model=model,
            max_tokens=2048,
            system=(
                "Design one surgical, gradient-free weight edit to Qwen3.5-0.8B for the "
                "user's goal. You can inspect MLP down-projection columns. Every column in "
                "an edit costs one parameter per row. Use zero-based column indices. "
                "Base your proposal on the tensor evidence and explain your hypothesis. "
                "Submit a proposal using propose_patch, or explain why you cannot propose one."
            ),
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        calls = [block for block in response.content if block.type == "tool_use"]
        if len(calls) != 1 or response.stop_reason != "tool_use":
            reason = " ".join(block.text for block in response.content if block.type == "text")
            raise ValueError(f"Claude did not submit a patch. {reason}".rstrip())
        call = calls[0]
        if call.name == "propose_patch":
            return call.input, response.model
        if call.name != "inspect_columns" or not remaining:
            raise ValueError("Claude requested an unavailable inspection.")
        remaining -= 1
        try:
            result = {"content": json.dumps(weights.inspect(call.input))}
        except (TypeError, ValueError) as exc:
            result = {"content": str(exc), "is_error": True}
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, **result},
        ]})
    raise ValueError("Claude exhausted the inspection budget without submitting a patch.")
