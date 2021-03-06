# Assets Module

A simple, secure module for dealing with fungible assets.

## Overview

The Assets module provides functionality for asset management of fungible asset classes
with a fixed supply, including:

- Asset Issuance
- Asset Transfer
- Asset Destruction
- Asset Burn
- Asset Mint

To use it in your runtime, you need to implement the assets [`Trait`](./trait.Trait.html).

The supported dispatchable functions are documented in the [`Call`](./enum.Call.html) enum.

### Terminology

- **Asset issuance:** The creation of a new asset, whose total supply will belong to the
  account that issues the asset.
- **Asset transfer:** The action of transferring assets from one account to another.
- **Asset mint** The action of minting assets for one's account
- **Asset burn** The action of burning assets for one's account
- **Asset destruction:** The process of an account removing its entire holding of an asset.
- **Fungible asset:** An asset whose units are interchangeable.
- **Non-fungible asset:** An asset for which each unit has unique characteristics.

### Goals

The assets system in Substrate is designed to make the following possible:

- Issue a unique asset to its creator's account.
- Move assets between accounts.
- Remove an account's balance of an asset when requested by that account's owner and update
  the asset's total supply.

## Interface

### Dispatchable Functions

- `issue` - Issues the total supply of a new fungible asset to the account of the caller of the function.
- `transfer` - Transfers an `amount` of units of fungible asset `id` from the balance of
  the function caller's account (`origin`) to a `target` account.
- `destroy` - Destroys the entire holding of a fungible asset `id` associated with the account
  that called the function.

Please refer to the [`Call`](./enum.Call.html) enum and its associated variants for documentation on each function.

### Public Functions

<!-- Original author of descriptions: @gavofyork -->

- `balance` - Get the asset `id` balance of `who`.
- `total_supply` - Get the total supply of an asset `id`.

Please refer to the [`Module`](./struct.Module.html) struct for details on publicly available functions.

## Usage

The following example shows how to use the Assets module in your runtime by exposing public functions to:

- Issue a new fungible asset for a token distribution event (airdrop).
- Query the fungible asset holding balance of an account.
- Query the total supply of a fungible asset that has been issued.

### Prerequisites

Import the Assets module and types and derive your runtime's configuration traits from the Assets module trait.

### Simple Code Snippet

```rust
use pallet_assets as assets;
use frame_support::{decl_module, dispatch, ensure};
use frame_system::ensure_signed;

pub trait Trait: assets::Trait { }

decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
		pub fn issue_token_airdrop(origin) -> dispatch::DispatchResult {
			let sender = ensure_signed(origin).map_err(|e| e.as_str())?;

			const ACCOUNT_ALICE: u64 = 1;
			const ACCOUNT_BOB: u64 = 2;
			const COUNT_AIRDROP_RECIPIENTS: u64 = 2;
			const TOKENS_FIXED_SUPPLY: u64 = 100;

			ensure!(!COUNT_AIRDROP_RECIPIENTS.is_zero(), "Divide by zero error.");

			let asset_id = Self::next_asset_id();

			<NextAssetId<T>>::mutate(|asset_id| *asset_id += 1);
			<Balances<T>>::insert((asset_id, &ACCOUNT_ALICE), TOKENS_FIXED_SUPPLY / COUNT_AIRDROP_RECIPIENTS);
			<Balances<T>>::insert((asset_id, &ACCOUNT_BOB), TOKENS_FIXED_SUPPLY / COUNT_AIRDROP_RECIPIENTS);
			<TotalSupply<T>>::insert(asset_id, TOKENS_FIXED_SUPPLY);

			Self::deposit_event(RawEvent::Issued(asset_id, sender, TOKENS_FIXED_SUPPLY));
			Ok(())
		}
	}
}
```

## Assumptions

Below are assumptions that must be held when using this module. If any of
them are violated, the behavior of this module is undefined.

- The total count of assets should be less than
  `Trait::AssetId::max_value()`.

## Related Modules

- [`System`](../frame_system/index.html)
- [`Support`](../frame_support/index.html)

License: Apache-2.0
