// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Literal module contains foundational types that are used to represent literals in DataFusion.

use crate::expr::FieldMetadata;
use crate::Expr;
use datafusion_common::ScalarValue;

/// Create a literal expression
pub fn lit<T: Literal>(n: T) -> Expr {
    n.lit()
}

pub fn lit_with_metadata<T: Literal>(n: T, metadata: Option<FieldMetadata>) -> Expr {
    let Some(metadata) = metadata else {
        return n.lit();
    };

    let Expr::Literal(sv, prior_metadata) = n.lit() else {
        unreachable!();
    };
    let new_metadata = match prior_metadata {
        Some(mut prior) => {
            prior.extend(metadata);
            prior
        }
        None => metadata,
    };

    Expr::Literal(sv, Some(new_metadata))
}

/// Create a literal timestamp expression
pub fn lit_timestamp_nano<T: TimestampLiteral>(n: T) -> Expr {
    n.lit_timestamp_nano()
}

/// Trait for converting a type to a [`Literal`] literal expression.
pub trait Literal {
    /// convert the value to a Literal expression
    fn lit(&self) -> Expr;
}

/// Trait for converting a type to a literal timestamp
pub trait TimestampLiteral {
    fn lit_timestamp_nano(&self) -> Expr;
}

impl Literal for &str {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::from(*self), None)
    }
}

impl Literal for String {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::from(self.as_ref()), None)
    }
}

impl Literal for &String {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::from(self.as_ref()), None)
    }
}

impl Literal for Vec<u8> {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Binary(Some((*self).to_owned())), None)
    }
}

impl Literal for &[u8] {
    fn lit(&self) -> Expr {
        Expr::Literal(ScalarValue::Binary(Some((*self).to_owned())), None)
    }
}

impl Literal for ScalarValue {
    fn lit(&self) -> Expr {
        Expr::Literal(self.clone(), None)
    }
}

macro_rules! make_literal {
    ($TYPE:ty, $SCALAR:ident, $DOC: expr) => {
        #[doc = $DOC]
        impl Literal for $TYPE {
            fn lit(&self) -> Expr {
                Expr::Literal(ScalarValue::$SCALAR(Some(self.clone())), None)
            }
        }
    };
}

macro_rules! make_nonzero_literal {
    ($TYPE:ty, $SCALAR:ident, $DOC: expr) => {
        #[doc = $DOC]
        impl Literal for $TYPE {
            fn lit(&self) -> Expr {
                Expr::Literal(ScalarValue::$SCALAR(Some(self.get())), None)
            }
        }
    };
}

macro_rules! make_timestamp_literal {
    ($TYPE:ty, $SCALAR:ident, $DOC: expr) => {
        #[doc = $DOC]
        impl TimestampLiteral for $TYPE {
            fn lit_timestamp_nano(&self) -> Expr {
                Expr::Literal(
                    ScalarValue::TimestampNanosecond(Some((self.clone()).into()), None),
                    None,
                )
            }
        }
    };
}

make_literal!(bool, Boolean, "literal expression containing a bool");
make_literal!(f32, Float32, "literal expression containing an f32");
make_literal!(f64, Float64, "literal expression containing an f64");
make_literal!(i8, Int8, "literal expression containing an i8");
make_literal!(i16, Int16, "literal expression containing an i16");
make_literal!(i32, Int32, "literal expression containing an i32");
make_literal!(i64, Int64, "literal expression containing an i64");
make_literal!(u8, UInt8, "literal expression containing a u8");
make_literal!(u16, UInt16, "literal expression containing a u16");
make_literal!(u32, UInt32, "literal expression containing a u32");
make_literal!(u64, UInt64, "literal expression containing a u64");

make_nonzero_literal!(
    std::num::NonZeroI8,
    Int8,
    "literal expression containing an i8"
);
make_nonzero_literal!(
    std::num::NonZeroI16,
    Int16,
    "literal expression containing an i16"
);
make_nonzero_literal!(
    std::num::NonZeroI32,
    Int32,
    "literal expression containing an i32"
);
make_nonzero_literal!(
    std::num::NonZeroI64,
    Int64,
    "literal expression containing an i64"
);
make_nonzero_literal!(
    std::num::NonZeroU8,
    UInt8,
    "literal expression containing a u8"
);
make_nonzero_literal!(
    std::num::NonZeroU16,
    UInt16,
    "literal expression containing a u16"
);
make_nonzero_literal!(
    std::num::NonZeroU32,
    UInt32,
    "literal expression containing a u32"
);
make_nonzero_literal!(
    std::num::NonZeroU64,
    UInt64,
    "literal expression containing a u64"
);

make_timestamp_literal!(i8, Int8, "literal expression containing an i8");
make_timestamp_literal!(i16, Int16, "literal expression containing an i16");
make_timestamp_literal!(i32, Int32, "literal expression containing an i32");
make_timestamp_literal!(i64, Int64, "literal expression containing an i64");
make_timestamp_literal!(u8, UInt8, "literal expression containing a u8");
make_timestamp_literal!(u16, UInt16, "literal expression containing a u16");
make_timestamp_literal!(u32, UInt32, "literal expression containing a u32");

#[cfg(test)]
mod test {
    use std::num::NonZeroU32;

    use super::*;
    use crate::expr_fn::col;

    #[test]
    fn test_lit_nonzero() {
        let expr = col("id").eq(lit(NonZeroU32::new(1).unwrap()));
        let expected = col("id").eq(lit(ScalarValue::UInt32(Some(1))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_lit_timestamp_nano() {
        let expr = col("time").eq(lit_timestamp_nano(10)); // 10 is an implicit i32
        let expected =
            col("time").eq(lit(ScalarValue::TimestampNanosecond(Some(10), None)));
        assert_eq!(expr, expected);

        let i: i64 = 10;
        let expr = col("time").eq(lit_timestamp_nano(i));
        assert_eq!(expr, expected);

        let i: u32 = 10;
        let expr = col("time").eq(lit_timestamp_nano(i));
        assert_eq!(expr, expected);
    }
}
