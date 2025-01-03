import ast
import dataclasses
import inspect
from functools import cached_property
from typing import Set

from phiml.backend import ML_LOGGER


def get_unchanged_cache(obj, updated_fields: Set[str]):
    if not hasattr(obj, '__dict__'):
        return {}
    cache = {k: v for k, v in obj.__dict__.items() if isinstance(getattr(type(obj), k, None), cached_property)}
    result = {}
    for k, v in cache.items():
        deps = get_dependencies(obj.__class__, getattr(type(obj), k, None))
        if deps.isdisjoint(updated_fields):
            result[k] = v
    return result


def get_dependencies(cls: type, cls_property) -> Set[str]:
    if hasattr(cls, '__phiml_dep__') and cls_property in cls.__phiml_dep__:
        return cls.__phiml_dep__[cls_property]
    if isinstance(cls_property, cached_property):
        method = cls_property.func
    elif isinstance(cls_property, property):
        method = cls_property.fget
    else:
        assert callable(cls_property) and hasattr(cls_property, '__qualname__'), f"Dependency resolver failed on {cls_property} of {cls.__name__}"
        method = cls_property
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__}")
    source_code = inspect.getsource(method)
    indent0 = len(source_code) - len(source_code.lstrip())
    source_code_top = "\n".join([line[indent0:] for line in source_code.split("\n")])
    tree = ast.parse(source_code_top)
    analyzer = MemberVariableAnalyzer()
    analyzer.visit(tree)
    direct_deps = analyzer.member_vars
    fields = set([f.name for f in dataclasses.fields(cls)])
    field_deps = direct_deps & fields
    for prop_dep in direct_deps - fields:
        if not hasattr(cls, prop_dep):  # may be a dynamic dim, such as vector
            if hasattr(cls, 'shape'):
                prop_dep = 'shape'
            else:
                field_deps.update(fields)  # automatic shape() depends on all data_fields
                continue
        if isinstance(getattr(cls, prop_dep), (property, cached_property)):
            field_deps.update(get_dependencies(cls, getattr(cls, prop_dep)))
        elif callable(getattr(cls, prop_dep)):
            field_deps.update(get_dependencies(cls, getattr(cls, prop_dep)))
    if not hasattr(cls, '__phiml_dep__'):
        cls.__phiml_dep__ = {cls_property: field_deps}
    else:
        cls.__phiml_dep__[cls_property] = field_deps
    return field_deps


class MemberVariableAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.member_vars = set()

    def visit_Attribute(self, node):
        # Check if the attribute is accessed via `self`
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            self.member_vars.add(node.attr)
        self.generic_visit(node)
