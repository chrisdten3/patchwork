import math
import re

import torch


DOWN_PROJECTION = re.compile(r"(?:.*\.)?layers\.\d+\.mlp\.down_proj\.weight")


class Weights:
    def __init__(self, model):
        self.tensors = {
            name: param for name, param in model.named_parameters()
            if DOWN_PROJECTION.fullmatch(name) and param.ndim == 2 and param.is_floating_point()
        }
        if not self.tensors:
            raise ValueError("No editable MLP down-projection matrices found.")
        self.catalog = {name: list(param.shape) for name, param in self.tensors.items()}
        self.budget = sum(param.numel() for param in model.parameters()) // 1000

    def columns(self, name, start, count):
        if not isinstance(name, str) or name not in self.tensors:
            raise ValueError("Select a tensor from the catalog.")
        matrix = self.tensors[name]
        if (type(start) is not int or type(count) is not int
                or start < 0 or count < 1 or start + count > matrix.shape[1]):
            raise ValueError(f"Column range is outside {name}.")
        return matrix, matrix[:, start:start + count]

    @torch.inference_mode()
    def inspect(self, request):
        if not isinstance(request, dict) or set(request) != {"tensor", "start", "count"}:
            raise ValueError("Supply a tensor, start, and count.")
        if type(request["count"]) is not int or request["count"] > 8:
            raise ValueError("Inspect at most eight columns at a time.")
        _, columns = self.columns(request["tensor"], request["start"], request["count"])
        values = columns.float()
        return {
            "l2_norms": values.norm(dim=0).tolist(),
            "first_eight_rows": values[:8].tolist(),
        }

    @torch.inference_mode()
    def apply(self, proposal):
        if not isinstance(proposal, dict) or set(proposal) != {"rationale", "edits"}:
            raise ValueError("A patch must contain a rationale and edits.")
        rationale, edits = proposal["rationale"], proposal["edits"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("A patch must include a rationale.")
        if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
            raise ValueError("A patch must contain one to eight edits.")

        touched = 0
        changed = 0
        staged = []
        ranges = {}
        for edit in edits:
            required = {"tensor", "start", "count", "operation", "strength"}
            if not isinstance(edit, dict) or not required <= edit.keys() <= required | {"reference"}:
                raise ValueError("Invalid patch edit fields.")
            name, start, count = edit["tensor"], edit["start"], edit["count"]
            matrix, columns = self.columns(name, start, count)
            end = start + count
            prior_ranges = ranges.setdefault(name, [])
            if any(start < prior_end and prior_start < end for prior_start, prior_end in prior_ranges):
                raise ValueError("Patch edits cannot overlap.")
            prior_ranges.append((start, end))
            touched += columns.numel()
            if touched > self.budget:
                raise ValueError(f"Patch exceeds the {self.budget:,}-parameter budget.")
            strength = edit["strength"]
            if type(strength) not in (int, float) or not 0 <= strength <= 2:
                raise ValueError("Strength must be between 0 and 2.")
            values = columns.float()
            if edit["operation"] == "scale" and "reference" not in edit:
                updated = values * strength
            elif edit["operation"] == "project_out" and "reference" in edit and strength <= 1:
                _, reference = self.columns(name, edit["reference"], 1)
                direction = reference[:, 0].float()
                norm = direction.norm()
                if norm.item() == 0 or not math.isfinite(norm.item()):
                    raise ValueError("Reference column must have a finite, nonzero norm.")
                direction = direction / norm
                updated = values - strength * direction[:, None] * (direction @ values)
            else:
                raise ValueError("Use scale, or project_out with a reference and strength between 0 and 1.")
            updated = updated.to(matrix.dtype)
            if not torch.isfinite(updated).all().item():
                raise ValueError("Edit would create non-finite weights.")
            changed += torch.count_nonzero(updated != columns).item()
            staged.append((columns, updated))

        if not changed:
            raise ValueError("The proposed patch changes no weights.")
        for columns, updated in staged:
            columns.copy_(updated)
        return {"parameters_touched": touched, "parameters_changed": changed}
