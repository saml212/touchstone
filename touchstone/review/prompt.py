"""The review agent's LLM interface: its five-tool surface (TOOLS) and system prompt (SYSTEM).

Kept apart from the state machine in `review.agent` so the wording and the tool schemas read on
their own. The tools are the only way the agent touches the dataset, the reviews table, or Harbor.
"""

from __future__ import annotations

TOOLS = [
    {"type": "function", "function": {
        "name": "list_trials",
        "description": "List trials to review in priority order. filter: unsure|disagree|"
                       "unreviewed|needs_review|all.",
        "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "read_trial",
        "description": "Read one trial: instruction, the trajectory in plain words, and each "
                       "criterion's description and score. Sets it as the current trial.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}, "trial": {"type": "string"}},
                       "required": ["task", "trial"]}}},
    {"type": "function", "function": {
        "name": "record_review",
        "description": "Record whether the human agreed with the verifier on this trial.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"}, "trial": {"type": "string"},
            "verdict": {"type": "string", "description": "agree or disagree"},
            "note": {"type": "string"}}, "required": ["task", "trial", "verdict"]}}},
    {"type": "function", "function": {
        "name": "propose_change",
        "description": "Read back a criterion change (edit/add/remove a check, a weight, a judge "
                       "line, or instruction/persona wording) without writing it. `change` is one "
                       "object or a list of them.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}, "change": {}},
                       "required": ["task", "change"]}}},
    {"type": "function", "function": {
        "name": "apply_change",
        "description": "Write the change, regrade the trial's job, and report the new reward and "
                       "any other trials that moved. always=true applies it to every task with the "
                       "same job.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"}, "change": {},
            "always": {"type": "boolean"}}, "required": ["task", "change"]}}},
]

SYSTEM = (
    "You are Touchstone's review agent. A product person is checking their AI agent's benchmark "
    "with you, by voice or text. Walk one trial at a time: call list_trials, then read_trial. When "
    "you present a trial, FIRST state the facts verbatim from read_trial: the verifier reward as a "
    "percentage, then each criterion with whether it passed or failed. ONLY THEN gloss what "
    "the user wanted and what the agent did in plain words, and ask whether they agree it passed. "
    "Never claim a pass or a fail the scores do not show; when the trajectory is empty, say the "
    "agent did nothing. On agree, call record_review (verdict 'agree'); on disagree, record_review "
    "(verdict 'disagree'), ask what should have counted, call propose_change and read it back, and "
    "only after they confirm call apply_change (always=true if the rule holds for every task "
    "like it); then say the new reward and anything else that moved. Speak in plain product "
    "language. Never mention file names, tables, or JSON unless they ask. Reply with your spoken "
    "message when you are not calling a tool."
)
