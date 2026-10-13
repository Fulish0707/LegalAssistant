"""Explicit bindings for versioned answer rubrics; legacy grading stays separate."""
from dataclasses import dataclass
from types import ModuleType

from pydantic import BaseModel

from . import answer_judge as v2, answer_judge_v3 as v3


@dataclass(frozen=True)
class AnswerRubric:
    name: str
    module: ModuleType
    initial_selection: bool

    @property
    def grade_schema(self) -> type[BaseModel]:
        return self.module.AnswerGradeV3 if self.name == "v3" else self.module.AnswerGradeV2


RUBRICS = {
    "v2": AnswerRubric("v2", v2, False),
    "v3": AnswerRubric("v3", v3, True),
}


def get_answer_rubric(name: str) -> AnswerRubric:
    try:
        return RUBRICS[name]
    except KeyError:
        raise ValueError(f"Unsupported versioned answer rubric: {name}") from None
