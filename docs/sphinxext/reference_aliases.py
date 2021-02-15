# add this to conf.py

from sphinx.addnodes import pending_xref
from sphinx.ext.intersphinx import missing_reference

# https://stackoverflow.com/questions/62293058/how-to-add-objects-to-sphinxs-global-index-or-cross-reference-by-alias


def replace_target_intersphinx(app, env, node, contnode):
    reftarget_replace = app.config.reftarget_replace
    target = node.get("reftarget", None)
    if target:
        for old, new in reftarget_replace.items():
            node["reftarget"] = target.replace(old, new)
    return missing_reference(app, env, node, contnode)


def replace_target(app, doctree):
    reftarget_replace = app.config.reftarget_replace
    pending_xrefs = doctree.traverse(condition=pending_xref)

    for node in pending_xrefs:
        for old, new in reftarget_replace.items():
            target = node.get("reftarget")
            if target:
                new_target = target.replace(old, new)
                node["reftarget"] = new_target


def setup(app):
    app.add_config_value("reftarget_replace", default={}, rebuild="env")
    app.connect("doctree-read", replace_target)
    app.connect("missing-reference", replace_target_intersphinx)
