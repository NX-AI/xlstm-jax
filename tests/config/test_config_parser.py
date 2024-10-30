from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp

from xlstm_jax.configs import ConfigDict


@dataclass(kw_only=True, frozen=False)
class A(ConfigDict):
    a: int = 3


@dataclass(kw_only=True, frozen=False)
class B(ConfigDict):
    b: A = field(default_factory=lambda: A())


@dataclass(kw_only=True, frozen=False)
class C(ConfigDict):
    c: B = field(default_factory=lambda: B())
    d: jnp.dtype = jnp.float32


@dataclass(kw_only=True, frozen=False)
class D(ConfigDict):
    d: Literal["a", "b"] = "a"


@dataclass(kw_only=True, frozen=False)
class E(ConfigDict):
    e: jnp.dtype = jnp.float32


@dataclass(kw_only=True, frozen=False)
class F(A):
    f: int = 5


@dataclass(kw_only=True, frozen=False)
class G(ConfigDict):
    a: A | None = None
    g: int = 5


@dataclass(kw_only=True, frozen=False)
class H(ConfigDict):
    h: tuple[int, int] = (1, 3)


def test_parse_dataclasses_simple():
    a1 = A(a=5)
    a2 = ConfigDict.from_dict(A, a1.to_dict())
    assert a1.a == a2.a


def test_parse_dataclasses_stupidencoding():
    a1 = A(a=5)
    a2 = ConfigDict.from_dict(A, "A(a=5)")
    assert a1.a == a2.a


def test_parse_dataclasses_nested():
    b1 = B(b=A(a=8))
    b2 = ConfigDict.from_dict(B, b1.to_dict())
    assert b1.b.a == b2.b.a


def test_parse_dataclasses_dtype():
    c1 = C(c=B(b=A(a=10)), d=jnp.bfloat16)
    c2 = ConfigDict.from_dict(C, c1.to_dict())
    assert c1.c.b.a == c2.c.b.a
    assert c2.d == jnp.bfloat16


def test_parse_dataclasses_literal():
    d1 = D(d="b")
    d2 = ConfigDict.from_dict(D, d1.to_dict())
    assert d1.d == d2.d


def test_parse_dataclasses_classattr():
    e1 = E(e=jnp.int16)
    e2 = ConfigDict.from_dict(E, {"e": "<class 'jax.numpy.int16'>"})
    assert e1.e == e2.e


def test_parse_dataclasses_inheritance():
    f1 = F(a=1, f=2)
    f2 = ConfigDict.from_dict(F, f1.to_dict())
    assert f1.a == f2.a
    assert f2.f == f2.f


def test_parse_dataclasses_uniontype():
    g1 = G(a=A(a=0), g=2)
    g2 = ConfigDict.from_dict(G, g1.to_dict())
    assert g2.a is not None
    assert g1.a.a == g2.a.a
    assert g2.g == g2.g


def test_parse_dataclasses_uniontypenone():
    g1 = G(a=None, g=2)
    g2 = ConfigDict.from_dict(G, g1.to_dict())
    assert g1.a is g2.a
    assert g2.g == g2.g


def test_parse_dataclasses_stupidencoding2():
    h1 = H(h=(4, 4))
    h2 = ConfigDict.from_dict(H, "H(h=(4,4))")
    assert h1.h == tuple(h2.h)


def test_parse_dataclasses_strnone():
    res = ConfigDict.from_dict(str | None, "abc")
    assert res == "abc"


def test_parse_dataclasses_dict():
    res = ConfigDict.from_dict(dict[str, int], {"a": 1})
    assert res["a"] == 1


def test_parse_dataclasses_union():
    res = ConfigDict.from_dict(tuple[int, int] | int | None, 1)
    assert res == 1
