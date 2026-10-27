"""The tool that opens a skill.

The system prompt carries only the names and the one-line descriptions; this is
how the agent gets the rest.  Reading instructions changes nothing on disk, so
it needs no permission — and it is the cheapest way to give a small model a long
procedure: the tokens are spent once, in the turn that needs them, instead of on
every request for the rest of the session.
"""

from __future__ import annotations

from typing import Any

from localcode.agent.tools.base import Tool, ToolContext, ToolResult

#: Where the agent loop leaves the library.
STATE_KEY = "skills"


class SkillTool(Tool):
    name = "skill"
    kind = "read"
    description = ("Wczytaj instrukcje ze skilla (lista skillów jest w systemowym prompcie). "
                   "Użyj, zanim zaczniesz zadanie, którego skill dotyczy.")
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "nazwa skilla z listy"},
        },
        "required": ["name"],
    }

    def title(self, args: dict[str, Any]) -> str:
        return f"Skill: {args.get('name', '')}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        library = ctx.state.get(STATE_KEY)
        if library is None:
            return ToolResult(False, "W tej sesji nie ma żadnych skillów.", "brak skillów")
        name = str(args.get("name") or "").strip()
        if not name:
            return ToolResult(False, "Podaj nazwę skilla.", "brak nazwy")
        skill = library.get(name)
        if skill is None:
            available = ", ".join(s.name for s in library.all()) or "—"
            return ToolResult(False, f"Nie ma skilla „{name}”. Dostępne: {available}.",
                              f"nieznany skill: {name}")
        # Which variant this session serves (``skills.variant_for``). ``getattr``
        # because a test double standing in for the library predates variants.
        variant = getattr(library, "variant", "full")
        header = f"# Skill: {skill.name}\n{skill.description_for(variant)}\n"
        extras = skill.files()
        if extras:
            # The files travel with the skill; naming them is what makes a
            # script or a template in that folder usable at all.
            listing = ", ".join(extras[:12])
            header += f"\nPliki obok SKILL.md ({skill.directory}): {listing}\n"
        body = skill.body_for(variant)
        out = f"{header}\n{body}"
        # The clamp the base class promises: a 20 000-character skill would
        # otherwise blow past what a 16k window allows for one tool result.
        limit = getattr(ctx.limits, "max_output_chars", 0) or 0
        if limit and len(out) > limit:
            out = out[:limit].rstrip() + "\n\n[…skrócone…]"
        return ToolResult(True, out, f"wczytano skill „{skill.name}”")
