// DO NOT EDIT! This file was auto-generated by crates/build/re_types_builder/src/codegen/rust/api.rs
// Based on "crates/store/re_types/definitions/rerun/testing/datatypes/fuzzy.fbs".

#![allow(unused_imports)]
#![allow(unused_parens)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::cloned_instead_of_copied)]
#![allow(clippy::map_flatten)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::new_without_default)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]

use ::re_types_core::external::arrow2;
use ::re_types_core::ComponentName;
use ::re_types_core::SerializationResult;
use ::re_types_core::{ComponentBatch, MaybeOwnedComponentBatch};
use ::re_types_core::{DeserializationError, DeserializationResult};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AffixFuzzer5 {
    pub single_optional_union: Option<crate::testing::datatypes::AffixFuzzer4>,
}

impl ::re_types_core::SizeBytes for AffixFuzzer5 {
    #[inline]
    fn heap_size_bytes(&self) -> u64 {
        self.single_optional_union.heap_size_bytes()
    }

    #[inline]
    fn is_pod() -> bool {
        <Option<crate::testing::datatypes::AffixFuzzer4>>::is_pod()
    }
}

impl<T: Into<Option<crate::testing::datatypes::AffixFuzzer4>>> From<T> for AffixFuzzer5 {
    fn from(v: T) -> Self {
        Self {
            single_optional_union: v.into(),
        }
    }
}

impl std::borrow::Borrow<Option<crate::testing::datatypes::AffixFuzzer4>> for AffixFuzzer5 {
    #[inline]
    fn borrow(&self) -> &Option<crate::testing::datatypes::AffixFuzzer4> {
        &self.single_optional_union
    }
}

impl std::ops::Deref for AffixFuzzer5 {
    type Target = Option<crate::testing::datatypes::AffixFuzzer4>;

    #[inline]
    fn deref(&self) -> &Option<crate::testing::datatypes::AffixFuzzer4> {
        &self.single_optional_union
    }
}

impl std::ops::DerefMut for AffixFuzzer5 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Option<crate::testing::datatypes::AffixFuzzer4> {
        &mut self.single_optional_union
    }
}

::re_types_core::macros::impl_into_cow!(AffixFuzzer5);

impl ::re_types_core::Loggable for AffixFuzzer5 {
    #[inline]
    fn arrow2_datatype() -> arrow2::datatypes::DataType {
        #![allow(clippy::wildcard_imports)]
        use arrow2::datatypes::*;
        DataType::Struct(std::sync::Arc::new(vec![Field::new(
            "single_optional_union",
            <crate::testing::datatypes::AffixFuzzer4>::arrow2_datatype(),
            true,
        )]))
    }

    fn to_arrow2_opt<'a>(
        data: impl IntoIterator<Item = Option<impl Into<::std::borrow::Cow<'a, Self>>>>,
    ) -> SerializationResult<Box<dyn arrow2::array::Array>>
    where
        Self: Clone + 'a,
    {
        #![allow(clippy::wildcard_imports)]
        #![allow(clippy::manual_is_variant_and)]
        use ::re_types_core::{Loggable as _, ResultExt as _};
        use arrow2::{array::*, datatypes::*};
        Ok({
            let (somes, data): (Vec<_>, Vec<_>) = data
                .into_iter()
                .map(|datum| {
                    let datum: Option<::std::borrow::Cow<'a, Self>> = datum.map(Into::into);
                    (datum.is_some(), datum)
                })
                .unzip();
            let bitmap: Option<arrow2::bitmap::Bitmap> = {
                let any_nones = somes.iter().any(|some| !*some);
                any_nones.then(|| somes.into())
            };
            StructArray::new(
                Self::arrow2_datatype(),
                vec![{
                    let (somes, single_optional_union): (Vec<_>, Vec<_>) = data
                        .iter()
                        .map(|datum| {
                            let datum = datum
                                .as_ref()
                                .map(|datum| datum.single_optional_union.clone())
                                .flatten();
                            (datum.is_some(), datum)
                        })
                        .unzip();
                    let single_optional_union_bitmap: Option<arrow2::bitmap::Bitmap> = {
                        let any_nones = somes.iter().any(|some| !*some);
                        any_nones.then(|| somes.into())
                    };
                    {
                        _ = single_optional_union_bitmap;
                        crate::testing::datatypes::AffixFuzzer4::to_arrow2_opt(
                            single_optional_union,
                        )?
                    }
                }],
                bitmap,
            )
            .boxed()
        })
    }

    fn from_arrow2_opt(
        arrow_data: &dyn arrow2::array::Array,
    ) -> DeserializationResult<Vec<Option<Self>>>
    where
        Self: Sized,
    {
        #![allow(clippy::wildcard_imports)]
        use ::re_types_core::{Loggable as _, ResultExt as _};
        use arrow2::{array::*, buffer::*, datatypes::*};
        Ok({
            let arrow_data = arrow_data
                .as_any()
                .downcast_ref::<arrow2::array::StructArray>()
                .ok_or_else(|| {
                    let expected = Self::arrow2_datatype();
                    let actual = arrow_data.data_type().clone();
                    DeserializationError::datatype_mismatch(expected, actual)
                })
                .with_context("rerun.testing.datatypes.AffixFuzzer5")?;
            if arrow_data.is_empty() {
                Vec::new()
            } else {
                let (arrow_data_fields, arrow_data_arrays) =
                    (arrow_data.fields(), arrow_data.values());
                let arrays_by_name: ::std::collections::HashMap<_, _> = arrow_data_fields
                    .iter()
                    .map(|field| field.name.as_str())
                    .zip(arrow_data_arrays)
                    .collect();
                let single_optional_union = {
                    if !arrays_by_name.contains_key("single_optional_union") {
                        return Err(DeserializationError::missing_struct_field(
                            Self::arrow2_datatype(),
                            "single_optional_union",
                        ))
                        .with_context("rerun.testing.datatypes.AffixFuzzer5");
                    }
                    let arrow_data = &**arrays_by_name["single_optional_union"];
                    crate::testing::datatypes::AffixFuzzer4::from_arrow2_opt(arrow_data)
                        .with_context("rerun.testing.datatypes.AffixFuzzer5#single_optional_union")?
                        .into_iter()
                };
                arrow2::bitmap::utils::ZipValidity::new_with_validity(
                    ::itertools::izip!(single_optional_union),
                    arrow_data.validity(),
                )
                .map(|opt| {
                    opt.map(|(single_optional_union)| {
                        Ok(Self {
                            single_optional_union,
                        })
                    })
                    .transpose()
                })
                .collect::<DeserializationResult<Vec<_>>>()
                .with_context("rerun.testing.datatypes.AffixFuzzer5")?
            }
        })
    }
}
