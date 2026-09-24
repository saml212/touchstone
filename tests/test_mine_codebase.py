from touchstone.mine import scan_codebase


def _build_tree(root):
    (root / "agent.py").write_text(
        'SYSTEM_PROMPT = "You are a helpful support agent."\n'
        'def run():\n'
        '    tools = [{"name": "lookup"}]\n'
        '    return tools\n',
        encoding="utf-8",
    )
    (root / "client.ts").write_text(
        "const tools = [\n"
        '  { name: "search", description: "search the web" },\n'
        "];\n",
        encoding="utf-8",
    )
    (root / "schema.json").write_text(
        '{\n  "name": "refund",\n  "parameters": {\n    "type": "object"\n  }\n}\n',
        encoding="utf-8",
    )
    (root / "anthropic_tool.py").write_text(
        'TOOL = {"name": "x", "input_schema": {"type": "object"}}\n',
        encoding="utf-8",
    )
    (root / "big.py").write_text("# junk\n" + ("x = 1\n" * 400_000), encoding="utf-8")
    nm = root / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text('const SYSTEM_PROMPT = "ignore me";\n', encoding="utf-8")


def test_scanner_finds_prompts_and_schemas_and_skips_junk(tmp_path):
    _build_tree(tmp_path)
    snippets = scan_codebase(tmp_path)
    kinds = {(s.path, s.kind) for s in snippets}

    assert ("agent.py", "system_prompt") in kinds
    assert ("agent.py", "tools") in kinds
    assert ("client.ts", "tools") in kinds
    assert ("schema.json", "parameters") in kinds
    assert ("anthropic_tool.py", "input_schema") in kinds

    paths = {s.path for s in snippets}
    assert "big.py" not in paths  # >1MB skipped
    assert not any("node_modules" in p for p in paths)  # vendored dir skipped


def test_scanner_captures_block_and_line(tmp_path):
    _build_tree(tmp_path)
    ts_tools = next(
        s for s in scan_codebase(tmp_path) if s.path == "client.ts" and s.kind == "tools"
    )
    assert ts_tools.line == 1
    assert "search" in ts_tools.text  # bracket block captured across lines


def test_scanner_sorted_and_deterministic(tmp_path):
    _build_tree(tmp_path)
    first = scan_codebase(tmp_path)
    second = scan_codebase(tmp_path)
    assert first == second
    assert first == sorted(first, key=lambda s: (s.path, s.line))


def test_scanner_on_single_file(tmp_path):
    f = tmp_path / "one.py"
    f.write_text('SYSTEM = "hi"\n', encoding="utf-8")
    snippets = scan_codebase(f)
    assert len(snippets) == 1 and snippets[0].kind == "system_prompt"


def test_scanner_empty_tree(tmp_path):
    assert scan_codebase(tmp_path) == []
