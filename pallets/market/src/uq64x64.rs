use crate::Trait;

// Decode a UQ64x64 as a u32
pub fn decode<T: Trait>(y: U64F64) -> T::Balance {
    T::Balance::From(y.to_num::<u32>())
}

// Encode a Balance to UQ64x64
pub fn encode<T: Trait>(y: T::Balance) -> U64F64 {
    U64F64::from_num(y);
}

// Divide a UQ64x64 by a Balance, returning a U64F64
pub fn uqdiv<T: Trait>(x: U64F64, y: T::Balance) -> U64F64 {
    x / U64F64::from_num(y);
}
