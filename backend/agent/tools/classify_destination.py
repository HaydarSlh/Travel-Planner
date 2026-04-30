"""classify_destination tool — wraps the scikit-learn Pipeline loaded at startup.

Inputs are the 12 destination features (validated by ClassifyDestinationInput).
Destination name and country are intentionally absent — they are identifiers
that would leak label information and prevent generalisation to new places.

Returns ClassifyDestinationOutput on success, ToolError on any failure.
Errors are structured data so the agent loop can feed them back to the LLM
for retry without crashing the graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from schemas.tools import ClassifyDestinationInput, ClassifyDestinationOutput, ToolError

_TOOL_NAME = "classify_destination"


def classify_destination(
    inp: ClassifyDestinationInput,
    pipeline,  # sklearn Pipeline — typed loosely to avoid importing sklearn at module level
    threshold: float,
) -> ClassifyDestinationOutput | ToolError:
    """Run the classifier pipeline and return style + confidence.

    The pipeline expects a single-row DataFrame whose columns match the 12
    features used during training (NUMERIC_COLS + CATEGORICAL_COLS order does
    not matter — ColumnTransformer selects by name).
    """
    try:
        row = pd.DataFrame([inp.model_dump()])
        proba: np.ndarray = pipeline.predict_proba(row)[0]
        classes: list[str] = list(pipeline.classes_)

        best_idx = int(np.argmax(proba))
        predicted_style = classes[best_idx]
        confidence = float(proba[best_idx])
        per_class_probs = {cls: round(float(p), 6) for cls, p in zip(classes, proba)}

        return ClassifyDestinationOutput(
            predicted_style=predicted_style,  # type: ignore[arg-type]
            confidence=confidence,
            per_class_probs=per_class_probs,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolError(tool=_TOOL_NAME, error=str(exc))
