"""The review agent's LLM interface: its five-tool surface (TOOLS) and system prompt (SYSTEM).

Kept apart from the state machine in `review.agent` so the wording and the tool schemas read on
their own. The tools are the only way the agent touches the dataset, the reviews table, or Harbor.
"""

from __future__ import annotations

# The exact shape of a criterion change, spelled out for the model with one example per op so it
# never invents its own keys (an {"action": ...} shape silently does nothing).
CHANGE_SCHEMA = (
    "`change` is ONE object (or a list of them) using EXACTLY these keys — never invent others. "
    "To change what a check expects (the usual case) keep everything else and set only the value: "
    "{\"op\":\"edit\",\"file\":\"<from the criterion handle>\",\"criterion\":<1-based index>,"
    "\"params\":{\"expected\": <new value, a number stays a number>}}. "
    "To remove a check: {\"op\":\"remove\",\"file\":\"...\",\"criterion\":<index>}. "
    "To add a check, pick its KIND from what the person said, then give the full call with a bare "
    "rewardkit name (never \"rk.\") and REAL values copied from the trial — never a placeholder "
    "like \"<...>\", an empty string, or \"TODO\". "
    "(1) They say the agent SHOULD (or should NOT) have USED/CALLED a tool → a trajectory check: "
    "{\"op\":\"add\",\"file\":\"tests/correctness/trajectory.py\",\"params\":{\"fn\":"
    "\"trajectory_tool_used\",\"args\":[\"<tool name, e.g. order_status>\"]}} (or "
    "\"trajectory_tool_not_used\"). NEVER a sqlite query for a tool-use ask. "
    "(2) They say a RECORD/FIELD should be a value → a state check, copying the db path and table "
    "from a sibling line in state.py or from read_trial's state_db_paths: {\"op\":\"add\",\"file\":"
    "\"tests/correctness/state.py\",\"params\":{\"fn\":\"sqlite_query_equals\",\"args\":["
    "\"simulators/<service>/state.db\",\"SELECT <col> FROM <table> WHERE <id>=<value>\","
    "<expected>]}}. (3) They say the REPLY should mention something → {\"op\":\"add\",\"file\":"
    "\"tests/correctness/answer.py\",\"params\":{\"fn\":\"file_contains\",\"args\":["
    "\"/app/output.json\",\"<text>\"]}}. Always include \"description\": one plain sentence a "
    "product person can say yes to. "
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
                       "criterion a stable `handle` \"<file>:<index>\" to pass to a change, and "
                       "`criteria_files` lists the task's criteria files — edit/remove must name "
                       "one of those; an add may name a new file and it is created. Each editable "
                       "entry carries the criterion's source line — copy a db path and table from "
                       "a sibling state check; `state_db_paths` lists this task's real simulator "
                       "databases to copy verbatim. "
                       "Sets it as the current trial. A needs_review task has no trajectory: this "
                       "returns `gate_failure` with a `reason` — tell the person that reason.",
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
        "description": "Apply the change you just proposed and read back, regrade the trial's job, "
                       "and report the new reward and any other trials that moved. Takes no change "
                       "argument — it always writes the last proposed change, never a new one. "
                       "always=true ONLY when the person said every/all/always/everywhere; "
                       "otherwise false.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"},
            "always": {"type": "boolean"}}}}},
]

SYSTEM = (
    "You are Touchstone's review agent. A product person is checking their AI agent's benchmark "
    "with you, by voice or text. Walk one trial at a time: call list_trials, then read_trial. When "
    "you present a trial, FIRST state the facts verbatim from read_trial: the verifier reward as a "
    "percentage, then each criterion with whether it passed or failed. ONLY THEN gloss what "
    "the user wanted and what the agent did in plain words, and ask whether they agree it passed. "
    "Never claim a pass or a fail the scores do not show; when the trajectory is empty, say the "
    "agent did nothing. If read_trial returns a gate_failure, the task was set aside by the gate "
    "and has no trajectory to review: tell the person its `reason` plainly (do not call it an "
    "error), and offer to fix the task's criteria if they think it should have passed. On agree, "
    "call record_review (verdict 'agree'); on disagree, record_review "
    "(verdict 'disagree'), ask what should have counted, call propose_change and read it back "
    "ONCE. When they disagree with a CHECK, propose a criterion change — edit, add, remove, or a "
    "weight — never an instruction rewrite: rewording the instruction moves no reward on the "
    "recorded trials. Rewrite the instruction (an op 'text' change) ONLY when they say the wording "
    "itself is wrong, and then say it changes what the customer asks, not how it is scored, so "
    "rewards will not move until the tasks are re-run. The very next affirmative from them — any "
    "yes, including their answer to 'everywhere or "
    "just this one?' — is the go-ahead: call apply_change immediately (always=true if it holds for "
    "every task like it). Do not read the change back a second time or ask again. If apply_change "
    "returns an error, tell them plainly what failed and ask for the check by number; never reply "
    "with filler. If propose_change returns an error, fix the values and try ONCE more; if you "
    "still cannot build a check with real values, say 'I could not turn that into a check I trust "
    "— tell me the exact value you expect, or we skip it' and apply nothing. "
    "Then say the new reward and anything else that moved. Speak in plain product "
    "language. Never mention file names, tables, or JSON unless they ask. Reply with your spoken "
    "message when you are not calling a tool."
)
