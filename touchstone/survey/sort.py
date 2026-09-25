"""Sort tools into those that run on their own and those that cross the network.

A tool that calls a service (non-empty `calls` in the map) crosses the network and needs a
simulator; a tool that calls nothing is copied verbatim into the environment. Cross-checked against
the recordings: tools invoked in recordings but missing from the map are flagged `unmapped`, and
mapped tools never seen in recordings are flagged `unused`. Pure function over map + tool events.
"""

from __future__ import annotations

from .recordings import ToolEvent, recorded_tool_names


def sort_tools(map_data: dict, events: list[ToolEvent]) -> dict:
    mapped = {t["name"]: t for t in map_data.get("tools", [])}
    recorded = recorded_tool_names(events)
    crosses = [name for name, tool in mapped.items() if tool.get("calls")]
    own = [name for name, tool in mapped.items() if not tool.get("calls")]
    return {
        "runs_on_its_own": sorted(own),
        "crosses_the_network": sorted(crosses),
        "unmapped": sorted(recorded - set(mapped)),
        "unused": sorted(set(mapped) - recorded),
    }
