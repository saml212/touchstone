"""The review agent's LLM interface: its five-tool surface (TOOLS) and system prompt (SYSTEM).

Kept apart from the state machine in `review.agent` so the wording and the tool schemas read on
their own. The tools are the only way the agent touches the dataset, the reviews table, or Harbor.
"""

from __future__ import annotations

# The exact shape of a criterion change, spelled out for the model with one example per op so it
# never invents its own keys (an {"action": ...} shape silently does nothing).
CHANGE_SCHEMA = (
    "`change` is ONE object (or a list of them) using EXACTLY these keys — never invent others. "
    "For a check in a tests/*.py file: "
    "{\"op\": \"edit\"|\"add\"|\"remove\", \"file\": \"<from the criterion handle>\", "
    "\"criterion\": <1-based index from the handle>, "
    "\"params\": {\"fn\": \"<rewardkit fn>\", \"args\": [...]}}. "
    "Examples — edit: {\"op\":\"edit\",\"file\":\"tests/correctness/state.py\",\"criterion\":2,"
    "\"params\":{\"fn\":\"sqlite_query_equals\","
    "\"args\":[\"s/state.db\",\"SELECT refunded\",200]}}; "
    "add: {\"op\":\"add\",\"file\":\"tests/correctness/state.py\","
    "\"params\":{\"fn\":\"sqlite_query_equals\","
    "\"args\":[\"s/state.db\",\"SELECT status\",\"done\"]}}; "
    "remove: {\"op\":\"remove\",\"file\":\"tests/correctness/state.py\",\"criterion\":3}. "
    "A dimension weight: {\"op\":\"edit\",\"file\":\"tests/reward.toml\",\"criterion\":\"safety\","
    "\"weight\":1.0}. Instruction/persona wording: "
    "{\"op\":\"text\",\"file\":\"instruction.md\",\"text\":\"<full new text>\"}."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "list_trials",
        "description": "List trials to review in priority order. filter: unsure|disagree|"
                       "unreviewed|needs_review|all.",
        "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "read_trial",
        "description": "Read one trial: instruction, the trajectory in plain words, and each "
                       "criterion's description and score. Its `editable` list gives each "
                       "criterion a stable `handle` \"<file>:<index>\" to pass to a change. "
                       "Sets it as the current trial.",
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
        "description": CHANGE_SCHEMA + " Reads the change back WITHOUT writing it; a malformed "
                       "change returns an error to fix and re-draft.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}, "change": {}},
                       "required": ["task", "change"]}}},
    {"type": "function", "function": {
        "name": "apply_change",
        "description": "Write a change (same shape as propose_change), regrade the trial's job, "
                       "and report the new reward and any other trials that moved. always=true "
                       "applies it to every task with the same job. Refuses a change that does "
                       "not validate.",
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
    "(verdict 'disagree'), ask what should have counted, call propose_change and read it back "
    "ONCE. The very next affirmative from them — any yes, including their answer to 'everywhere or "
    "just this one?' — is the go-ahead: call apply_change immediately (always=true if it holds for "
    "every task like it). Do not read the change back a second time or ask again. If apply_change "
    "returns an error, tell them plainly what failed and ask for the check by number; never reply "
    "with filler. Then say the new reward and anything else that moved. Speak in plain product "
    "language. Never mention file names, tables, or JSON unless they ask. Reply with your spoken "
    "message when you are not calling a tool."
)
