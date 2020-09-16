// This file is part of Substrate.

// Copyright (C) Hyungsuk Kang
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Market Module
//!
//! A simple, secure module for exchanging with fungible assets in decentralized manner.
//!
//! ## Overview
//!
//! The Market module provides functionality for asset management of fungible asset classes
//! with a fixed supply, including:
//!
//! * Liquidity provider token issuance
//! * Compensation for providing liquidity
//! * Automated liquidity provisioning
//! * Asset exchange
//!
//! To use it in your runtime, you need to implement the market [`Trait`](./trait.Trait.html).
//!
//! The supported dispatchable functions are documented in the [`Call`](./enum.Call.html) enum.
//!
//! ### Terminology
//!
//! * **Liquidity provider token:** The creation of a new asset by providing liquidity between two fungible assets. Liquidity provider token act as the share of the pool and gets the profit created from exchange fee.
//! * **Asset exchange:** The process of an account transferring an asset to exchange with other kind of fungible asset.
//! * **Fungible asset:** An asset whose units are interchangeable.
//! * **Non-fungible asset:** An asset for which each unit has unique characteristics.
//!
//! ### Goals
//!
//! The market system in Substrate is designed to make the following possible:
//!
//! * Issue a liquidity provider token by depositing two different fungible assets.
//! * Swap assets between accounts with automated market price equation(e.g. X*Y=K or curve function from Kyber, dodoex, etc).
//! * Compensate liquidity provider
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! * `create_pair` - Issues the total supply of a new fungible asset to the account of the caller of the function.
//! * `swap` - Transfers an `amount` of units of fungible asset `id` from the balance of the function caller's account (`origin`) and receive the fungible asset to swap to
//!  to a `target` account.
//! * `burn` - Burns the lptoken and withdraws two fungible assets
//! * `mint` - Deposits two fungible assets and receive lp token
//! that called the function.
//!
//! Please refer to the [`Call`](./enum.Call.html) enum and its associated variants for documentation on each function.
//!
//! ### Public Functions
//! <!-- Original author of descriptions: @gavofyork -->
//!
//! * `reserves` - Get the reserves of two fungible assets in a given pair
//! * `pair` - Get the two fungible asset ids for a pair with a given liquidity asset id.
//! * `lpt` - Get the liquidity asset id from the two fungible asset ids
//!
//! Please refer to the [`Module`](./struct.Module.html) struct for details on publicly available functions.
//!
//! ## Usage
//!
//! The following example shows how to use the Assets module in your runtime by exposing public functions to:
//!
//! * Issue a new fungible asset for a token distribution event (airdrop).
//! * Query the fungible asset holding balance of an account.
//! * Query the total supply of a fungible asset that has been issued.
//!
//! ### Prerequisites
//!
//! Import the Assets module and types and derive your runtime's configuration traits from the Assets module trait.
//!
//! ### Simple Code Snippet
//!
//! ```rust,ignore
//! use pallet_market as market;
//! use frame_support::{decl_module, dispatch, ensure};
//! use frame_system::ensure_signed;
//!
//! pub trait Trait: assets::Trait + market::Trait { }
//!
//! decl_module! {
//! 	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
//! 		pub fn issue_liquidity_token(origin, token0: T::AssetId, amount0: T::Balance, token1: T::AssetId, amount1: T::Balance) -> dispatch::DispatchResult {
//! 			let sender = ensure_signed(origin).map_err(|e| e.as_str())?;
//!
//! 			const ACA: u64 = 1;
//! 			const aUSD: u64 = 2;
//!
//! 			ensure!(!COUNT_AIRDROP_RECIPIENTS.is_zero(), "Divide by zero error.");
//! 			<asset::Module<T>>::
//!
//! 			let asset_id = Self::next_asset_id();
//!
//! 			<NextAssetId<T>>::mutate(|asset_id| *asset_id += 1);
//! 			<Balances<T>>::insert((asset_id, &ACCOUNT_ALICE), TOKENS_FIXED_SUPPLY / COUNT_AIRDROP_RECIPIENTS);
//! 			<Balances<T>>::insert((asset_id, &ACCOUNT_BOB), TOKENS_FIXED_SUPPLY / COUNT_AIRDROP_RECIPIENTS);
//! 			<TotalSupply<T>>::insert(asset_id, TOKENS_FIXED_SUPPLY);
//!
//! 			Self::deposit_event(RawEvent::CreatePair(ACA, aUSD, ACA_aUSD_lpt));
//! 			Ok(())
//! 		}
//! 	}
//! }
//! ```
//!
//! ## Assumptions
//!
//! Below are assumptions that must be held when using this module.  If any of
//! them are violated, the behavior of this module is undefined.
//!
//! * The total count of assets should be less than
//!   `Trait::AssetId::max_value()`.
//!
//! ## Related Modules
//!
//! * [`System`](../frame_system/index.html)
//! * [`Support`](../frame_support/index.html)

// Ensure we're `no_std` when compiling for Wasm.
#![cfg_attr(not(feature = "std"), no_std)]

use asset;
/// Edit this file to define custom logic or remove it if it is not needed.
/// Learn more about FRAME and the core library of Substrate FRAME pallets:
/// https://substrate.dev/docs/en/knowledgebase/runtime/frame
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, traits::Get};
use frame_system::ensure_signed;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

/// Configure the pallet by specifying the parameters and types on which it depends.
pub trait Trait: frame_system::Trait + asset::Trait {
	/// Because this pallet emits events, it depends on the runtime's definition of an event.
	type Event: From<Event<Self>>
		+ Into<<Self as frame_system::Trait>::Event>
		+ Into<<Self as asset::Trait>::Event>;
}

// The pallet's runtime storage items.
// https://substrate.dev/docs/en/knowledgebase/runtime/storage
decl_storage! {
	// A unique name is used to ensure that the pallet's storage items are isolated.
	// This name may be updated, but each pallet in the runtime must use a unique name.
	// ---------------------------------vvvvvvvvvvvvvv
	trait Store for Module<T: Trait> as SwapModule {
		// Learn more about declaring storage items:
		// https://substrate.dev/docs/en/knowledgebase/runtime/storage#declaring-storage-items
		Something get(fn something): Option<u32>;
		pub Reserves get(fn reserves): map hasher(blake2_128_concat) T::AssetId => (T::Balance, T::Balance);
		pub Pair get(fn pair): map hasher(blake2_128_concat) T::AssetId => (T::AssetId, T::AssetId);
		pub LPTokens get(fn lpt): map hasher(blake2_128_concat) (T::AssetId, T::AssetId) => T::AssetId;
	}
}

// Pallets use events to inform users when important changes are made.
// https://substrate.dev/docs/en/knowledgebase/runtime/events
decl_event!(
	pub enum Event<T>
	where
		AccountId = <T as frame_system::Trait>::AccountId,
		Token0 = <T as asset::Trait>::AssetId,
		Token1 = <T as asset::Trait>::AssetId,
		LPToken = <T as asset::Trait>::AssetId,
		Balance = <T as asset::Trait>::Balance,
	{
		/// Event documentation should end with an array that provides descriptive names for event
		/// parameters. [something, who]
		SomethingStored(u32, AccountId),
		CreatePair(Token0, Token1, LPToken),
		Swap(Token0, Balance, Token1, Balance),
		MintedLiquidity(Token0, Token1, LPToken),
		BurnedLiquidity(LPToken, Token0, Token1),
	}
);

// Errors inform users that something went wrong.
decl_error! {
	pub enum Error for Module<T: Trait> {
		/// Error names should be descriptive.
		NoneValue,
		/// Errors should have helpful documentation associated with them.
		NotTheCreator,
		NotApproved,
	}
}

// Dispatchable functions allows users to interact with the pallet and invoke state changes.
// These functions materialize as "extrinsics", which are often compared to transactions.
// Dispatchable functions must be annotated with a weight and must return a DispatchResult.
decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
		// Errors must be initialized if they are used by the pallet.
		type Error = Error<T>;

		// Events must be initialized if they are used by the pallet.
		fn deposit_event() = default;

		/// An example dispatchable that takes a singles value as a parameter, writes the value to
		/// storage and emits an event. This function must be dispatched by a signed extrinsic.
		#[weight = 10_000 + T::DbWeight::get().writes(1)]
		pub fn do_something(origin, something: u32) -> dispatch::DispatchResult {
			// Check that the extrinsic was signed and get the signer.
			// This function will return an error if the extrinsic is not signed.
			// https://substrate.dev/docs/en/knowledgebase/runtime/origin
			let who = ensure_signed(origin)?;

			// Update storage.
			Something::put(something);

			// Emit an event.
			Self::deposit_event(RawEvent::SomethingStored(something, who));
			// Return a successful DispatchResult
			Ok(())
		}

		/// An example dispatchable that may throw a custom error.
		#[weight = 10_000 + T::DbWeight::get().reads_writes(1,1)]
		pub fn cause_error(origin) -> dispatch::DispatchResult {
			let _who = ensure_signed(origin)?;

			// Read a value from storage.
			match Something::get() {
				// Return an error if the value has not been set.
				None => Err(Error::<T>::NoneValue)?,
				Some(old) => {
					// Increment the value read from storage; will error in the event of overflow.
					let new = old.checked_add(1).ok_or(Error::<T>::StorageOverflow)?;
					// Update the value in storage with the incremented result.
					Something::put(new);
					Ok(())
				},
			}
		}

		#[weight = 10_000 + T::DbWeight::get().reads_writes(1,1)]
		pub fn create_pair(origin, token0: T::AssetId, token1: T::AssetId) -> dispatch::DispatchResult {
			let _who = ensure_signed(origin)?;
			Ok(())
		}

		#[weight = 10_000 + T::DbWeight::get().reads_writes(1,1)]
		pub fn swap(origin, from: T::AssetId, amount: T::Balance, to: T::AssetId) -> dispatch::DispatchResult {
			let _who = ensure_signed(origin)?;
			Ok(())
		}
	}
}
