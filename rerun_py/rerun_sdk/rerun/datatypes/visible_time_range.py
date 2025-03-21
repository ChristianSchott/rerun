# DO NOT EDIT! This file was auto-generated by crates/build/re_types_builder/src/codegen/python/mod.rs
# Based on "crates/store/re_types/definitions/rerun/datatypes/visible_time_range.fbs".

# You can extend this class by creating a "VisibleTimeRangeExt" class in "visible_time_range_ext.py".

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import pyarrow as pa
from attrs import define, field

from .. import datatypes
from .._baseclasses import (
    BaseBatch,
)
from .visible_time_range_ext import VisibleTimeRangeExt

__all__ = ["VisibleTimeRange", "VisibleTimeRangeArrayLike", "VisibleTimeRangeBatch", "VisibleTimeRangeLike"]


def _visible_time_range__timeline__special_field_converter_override(x: datatypes.Utf8Like) -> datatypes.Utf8:
    if isinstance(x, datatypes.Utf8):
        return x
    else:
        return datatypes.Utf8(x)


@define(init=False)
class VisibleTimeRange(VisibleTimeRangeExt):
    """**Datatype**: Visible time range bounds for a specific timeline."""

    # __init__ can be found in visible_time_range_ext.py

    timeline: datatypes.Utf8 = field(converter=_visible_time_range__timeline__special_field_converter_override)
    # Name of the timeline this applies to.
    #
    # (Docstring intentionally commented out to hide this field from the docs)

    range: datatypes.TimeRange = field()
    # Time range to use for this timeline.
    #
    # (Docstring intentionally commented out to hide this field from the docs)


VisibleTimeRangeLike = VisibleTimeRange
VisibleTimeRangeArrayLike = Union[
    VisibleTimeRange,
    Sequence[VisibleTimeRangeLike],
]


class VisibleTimeRangeBatch(BaseBatch[VisibleTimeRangeArrayLike]):
    _ARROW_DATATYPE = pa.struct([
        pa.field("timeline", pa.utf8(), nullable=False, metadata={}),
        pa.field(
            "range",
            pa.struct([
                pa.field(
                    "start",
                    pa.dense_union([
                        pa.field("_null_markers", pa.null(), nullable=True, metadata={}),
                        pa.field("CursorRelative", pa.int64(), nullable=False, metadata={}),
                        pa.field("Absolute", pa.int64(), nullable=False, metadata={}),
                        pa.field("Infinite", pa.null(), nullable=True, metadata={}),
                    ]),
                    nullable=True,
                    metadata={},
                ),
                pa.field(
                    "end",
                    pa.dense_union([
                        pa.field("_null_markers", pa.null(), nullable=True, metadata={}),
                        pa.field("CursorRelative", pa.int64(), nullable=False, metadata={}),
                        pa.field("Absolute", pa.int64(), nullable=False, metadata={}),
                        pa.field("Infinite", pa.null(), nullable=True, metadata={}),
                    ]),
                    nullable=True,
                    metadata={},
                ),
            ]),
            nullable=False,
            metadata={},
        ),
    ])

    @staticmethod
    def _native_to_pa_array(data: VisibleTimeRangeArrayLike, data_type: pa.DataType) -> pa.Array:
        from rerun.datatypes import TimeRangeBatch, Utf8Batch

        if isinstance(data, VisibleTimeRange):
            data = [data]

        return pa.StructArray.from_arrays(
            [
                Utf8Batch([x.timeline for x in data]).as_arrow_array(),  # type: ignore[misc, arg-type]
                TimeRangeBatch([x.range for x in data]).as_arrow_array(),  # type: ignore[misc, arg-type]
            ],
            fields=list(data_type),
        )
