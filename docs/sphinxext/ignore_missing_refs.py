# -*- coding: utf-8 -*-
from docutils import nodes

PACKAGES = ["kartothek"]


def _is_external_target(target):
    return not any(((target == p) or target.startswith(p + ".") for p in PACKAGES))


def _is_private_target(target):
    return any((part.startswith("_") for part in target.split(".")))


def missing_reference(app, env, node, contnode):
    target = node["reftarget"]
    if _is_external_target(target) or _is_private_target(target):
        newnode = nodes.reference("", "", internal=False, refuri="#", reftitle="")
        newnode.append(contnode)
        return newnode


def setup(app):
    app.connect("missing-reference", missing_reference)
    return {"version": "0.1", "parallel_read_safe": True}
