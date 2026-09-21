"""Structured schema for parallel constrained scoring (jev-rlcd Route A)."""

from dataclasses import dataclass, field


@dataclass
class FieldDefinition:
    name: str
    field_type: str  # "enum" | "boolean"
    description: str
    choices: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.field_type == "boolean":
            self.choices = ["true", "false"]
        assert self.choices and len(self.choices) >= 2, self.name


class StructuredSchema:
    def __init__(self, fields: dict[str, dict] | dict[str, FieldDefinition]):
        self.fields: list[FieldDefinition] = []
        for name, spec in fields.items():
            if isinstance(spec, FieldDefinition):
                fd = spec
            else:
                spec = dict(spec)
                if "type" in spec:
                    spec["field_type"] = spec.pop("type")
                fd = FieldDefinition(name=name, **spec)
            self.fields.append(fd)

    def __len__(self):
        return len(self.fields)

    def catalog_text(self) -> str:
        """All field descriptions + choices, shown once in the shared prompt."""
        lines = []
        for f in self.fields:
            if f.field_type == "boolean":
                lines.append(f"- {f.name} (boolean): {f.description}")
            else:
                lines.append(f"- {f.name} (one of: {', '.join(f.choices)}): {f.description}")
        return "\n".join(lines)
