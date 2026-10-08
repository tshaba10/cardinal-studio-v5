"""Upload parser's intermediate records; NOT the exported generator contract."""

from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Element(StrictModel):
    col_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: Literal["unknown", "string", "int16", "decimal", "date", "timestamp", "boolean"]
    nullable: bool
    null_rate: float = Field(ge=0, le=1)
    unique: bool
    # Observed sample frequencies aid the interview; they are not generation rules.
    values: list[str | int | float | bool] = Field(max_length=100)
    weights: list[float] = Field(max_length=100)
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_values(self):
        if not self.nullable and self.null_rate != 0:
            raise ValueError("Non-nullable elements must have null_rate=0.")
        if len(self.values) != len(self.weights) or any(w < 0 for w in self.weights):
            raise ValueError("Each value needs one nonnegative weight.")
        # Header-only uploads are valid context; the interview supplies their meaning.
        if not self.values:
            return self
        if abs(sum(self.weights) - 1) > 0.000001:
            raise ValueError("Weights must sum to 1.")
        if len({(type(v).__name__, str(v)) for v in self.values}) != len(self.values):
            raise ValueError("Allowed values must not contain duplicates.")
        for value in self.values:
            if self.type == "boolean" and type(value) is not bool:
                raise ValueError("Boolean values must be true or false.")
            if self.type == "int16" and (type(value) is not int or not -32768 <= value <= 32767):
                raise ValueError("int16 values must be integers between -32768 and 32767.")
            if self.type == "decimal" and type(value) not in (int, float):
                raise ValueError("Decimal values must be numbers.")
            if self.type in ("string", "date", "timestamp") and type(value) is not str:
                raise ValueError("Text and date values must be strings.")
            if self.type == "date":
                date.fromisoformat(value)
            if self.type == "timestamp" and datetime.fromisoformat(value).tzinfo is None:
                raise ValueError("Timestamps need a timezone offset.")
        return self


class Requirements(StrictModel):
    elements: list[Element] = Field(default_factory=list, max_length=50)
    assumptions: list[str] = Field(default_factory=list)
