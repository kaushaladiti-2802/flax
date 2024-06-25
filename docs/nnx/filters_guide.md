---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Using Filters

Filters are used extensively in NNX as a way to create `State` groups in APIs
such as `nnx.split` and `nnx.state`. For example:

```{code-cell}
from flax import nnx

class Foo(nnx.Module):
  def __init__(self):
    self.a = nnx.Param(0)
    self.b = nnx.BatchStat(True)

foo = Foo()

graphdef, params, batch_stats = nnx.split(foo, nnx.Param, nnx.BatchStat)

print(f'{params = }')
print(f'{batch_stats = }')
```

Here `nnx.Param` and `nnx.BatchStat` are used as Filters to split the model into two groups: one with the parameters and the other with the batch statistics. However this begs the questions:

* What is a Filter?
* Why are types, such as `Param` or `BatchStat`, Filters?
* How is `State` grouped / filtered?

+++

## The Filter Protocol

In general Filter are predicate functions of the form:

```python

def f(path: tuple[Key, ...], value: Any) -> bool:

```
where `Key` is a hashable and comparable type, `path` is a tuple of `Key`s representing the path to the value in a nested structure, and `value` is the value at the path. The function returns `True` if the value should be included in the group and `False` otherwise.

Types are obviously not functions. The reason why types are used as Filters is because internally they are converted to functions. For example, `Param` is roughly converted to a function like this:

```{code-cell}
def is_param(path, value) -> bool:
  return isinstance(value, nnx.Param) or (
    hasattr(value, 'type') and issubclass(value.type, nnx.Param)
  )
```

Such a function matches any value that is an instance of `Param` or any value that hash a `type` attribute that is an instance of a subclass of `Param`. Internally NNX uses `OfType` which creates a callable of the form for a given type.

```{code-cell}
is_param = nnx.OfType(nnx.Param)

print(f'{is_param((), nnx.Param(0)) = }')
print(f'{is_param((), nnx.VariableState(type=nnx.Param, value=0)) = }')
```

## The Filter DSL

To avoid users having to create these functions, NNX exposes a small
DSL, formalized as the `nnx.filterlib.Filter` type, which lets users for common cases
pass certain non-callable types such as types, booleans, ellipsis, tuples/lists, etc,
and converts them to the appropriate callable internally.

Here is a list of all the callable Filters included in NNX and their DSL literals (when available):


| Literal | Callable | Description |
|--------|----------------------|-------------|
| `...` | `Everything()` | Matches all values |
| `None` | `Nothing()` | Matches no values |
| `True` | `Everything()` | Matches all values |
| `False` | `Nothing()` | Matches no values |
| `type` | `OfType(type)` | Matches values that are instances of `type` or have a `type` attribute that is an instance of `type` |
| | `PathContains(key)` | Matches values that have an associated `path` that contains the given `key` |
| `'{filter}'` | `WithTag('{filter}')` | Matches values that have string `tag` attribute equal to `'{filter}'` |
| `(*filters)` | `Any(*filters)` | Matches values that match any of the inner `filters` |
| `[*filters]` | `Any(*filters)` | Matches values that match all of the inner `filters` |
| | `All(*filters)` | Matches values that match all of the inner `filters` |
| | `Not(filter)` | Matches values that do not match the inner `filter` |

If you wish to manually create a callable from a literal to can use `nnx.filterlib.to_predicate`:

```{code-cell}
is_param = nnx.filterlib.to_predicate(nnx.Param)
everything = nnx.filterlib.to_predicate(...)
nothing = nnx.filterlib.to_predicate(False)
any = nnx.filterlib.to_predicate([nnx.Param, 'dropout'])

print(f'{type(is_param) = }')
print(f'{type(everything) = }')
print(f'{type(nothing) = }')
print(f'{type(any) = }')
```

## Grouping States

With the knowledge of Filters at hand, let's see how `nnx.split` is roughly implemented. Key ideas:

* Use `nnx.graph.flatten` to get the `GraphDef` and `State` representation of the node.
* Convert all the filters to predicates.
* Use `State.flat_state` to get the flat representation of the state.
* Traverse all the `(path, value)` pairs in the flat state and group them according to the filters.
* Use `State.from_flat_state` to convert the flat states to nested `State`s.

```{code-cell}
from typing import Any
KeyPath = tuple[nnx.graph.Key, ...]

def split(node, *filters):
  graphdef, state, _ = nnx.graph.flatten(node)
  predicates = [nnx.filterlib.to_predicate(f) for f in filters]
  flat_states: list[dict[KeyPath, Any]] = [{} for p in predicates]

  for path, value in state.flat_state().items():
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i][path] = value
        break
    else:
      raise ValueError(f'No filter matched {path = } {value = }')
    
  states: tuple[nnx.GraphState, ...] = tuple(
    nnx.State.from_flat_path(flat_state) for flat_state in flat_states
  )
  return graphdef, *states

# lets test it...
foo = Foo()

graphdef, params, batch_stats = split(foo, nnx.Param, nnx.BatchStat)

print(f'{params = }')
print(f'{batch_stats = }')
```

One very important thing to note is that **the order of the filters is important**. The first filter that matches a value will be the group that the value is placed in, therefore you should place more specific filters before more general filters. For example if we create a `SpecialParam` type that is a subclass of `Param` and create an object `Bar` that contains both types of parameters, if we try to split the `Param`s before the `SpecialParam`s, all the values will be placed in the `Param` group and the `SpecialParam` group will be empty because all `SpecialParam`s are also `Param`s:

```{code-cell}
class SpecialParam(nnx.Param):
  pass

class Bar(nnx.Module):
  def __init__(self):
    self.a = nnx.Param(0)
    self.b = SpecialParam(0)

bar = Bar()

graphdef, params, special_params = split(bar, nnx.Param, SpecialParam) # wrong!
print(f'{params = }')
print(f'{special_params = }')
```

Reversing the order will make sure that the `SpecialParam` are captured first

```{code-cell}
graphdef, special_params, params = split(bar, SpecialParam, nnx.Param) # correct!
print(f'{params = }')
print(f'{special_params = }')
```

```{code-cell}

```
