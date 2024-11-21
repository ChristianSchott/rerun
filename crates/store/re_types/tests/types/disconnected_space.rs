use std::collections::HashMap;

use re_types::{archetypes::DisconnectedSpace, Archetype as _, AsComponents as _};

use crate::util;

#[test]
fn roundtrip() {
    let all_expected = [
        DisconnectedSpace {
            disconnected_space: true.into(),
        }, //
        DisconnectedSpace {
            disconnected_space: false.into(),
        },
    ];

    let all_arch = [
        DisconnectedSpace::new(true),  //
        DisconnectedSpace::new(false), //
    ];

    let expected_extensions: HashMap<_, _> = [
        (
            "disconnected_space",
            vec!["rerun.components.DisconnectedSpace"],
        ), //
        (
            "disconnected_space",
            vec!["rerun.components.DisconnectedSpace"],
        ), //
    ]
    .into();

    for (expected, arch) in all_expected.into_iter().zip(all_arch) {
        similar_asserts::assert_eq!(expected, arch);

        eprintln!("arch = {arch:#?}");
        let serialized = arch.to_arrow2().unwrap();
        for (field, array) in &serialized {
            // NOTE: Keep those around please, very useful when debugging.
            // eprintln!("field = {field:#?}");
            // eprintln!("array = {array:#?}");
            eprintln!("{} = {array:#?}", field.name);

            // TODO(cmc): Re-enable extensions and these assertions once `arrow2-convert`
            // has been fully replaced.
            if false {
                util::assert_extensions(
                    &**array,
                    expected_extensions[field.name.as_str()].as_slice(),
                );
            }
        }

        let deserialized = DisconnectedSpace::from_arrow2(serialized).unwrap();
        similar_asserts::assert_eq!(expected, deserialized);
    }
}
