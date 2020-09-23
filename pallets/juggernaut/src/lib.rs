#![cfg_attr(not(feature = "std"), no_std)]
use engine;

/// Edit this file to define custom logic or remove it if it is not needed.
/// Learn more about FRAME and the core library of Substrate FRAME pallets:
/// https://substrate.dev/docs/en/knowledgebase/runtime/frame

use frame_support::{decl_module, decl_storage, decl_event,ensure, decl_error, dispatch, traits::Get};
use frame_system::ensure_signed;
use frame_support::codec::{Encode, Decode};

use engine::nl::NeuralLayer;
use engine::nn::NeuralNetwork;
use engine::activation::{
	HyperbolicTangent,
	Sigmoid,
	Identity,
	LeakyRectifiedLinearUnit,
	RectifiedLinearUnit,
	SoftMax,
	SoftPlus	
};

use engine::sample::Sample;



#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;



/// Configure the pallet by specifying the parameters and types on which it depends.
pub trait Trait: frame_system::Trait {
	/// Because this pallet emits events, it depends on the runtime's definition of an event.
	type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
}

type NeuralKey<AcId> = (AcId,String); 

#[derive(Encode, Decode,Default, Clone, PartialEq,Hash)]
pub struct NeuralStruct {
	name: String,
	neural_network: Option<String>
}

impl NeuralStruct{
	pub fn new(name: String)-> Self{
		NeuralStruct{
			name: name,
			neural_network: None
		}
	}
	pub fn get_model(&self) -> Result<NeuralNetwork,String>{
		if let Some(ref nn)=self.neural_network{
			NeuralNetwork::get_neural_from_str(nn.clone())
		}else{
			Err("No Model".to_string())
		}
	} 
	pub fn add_layers(&mut self,layer: NeuralLayer){
		if let Ok(mut net) = self.get_model(){
			net.add_layer(layer);
			self.neural_network=Some(net.to_string());
		}	
	}
	pub fn get_model_string(&self)->String{
		self.neural_network.as_ref().unwrap_or(&"".to_owned()).to_string()
	}
}

decl_storage! {
	trait Store for Module<T: Trait> as TemplateModule{
		NeuralContainer get(fn neural_container): map hasher(blake2_128_concat) (T::AccountId,String) => NeuralStruct;
		DataContainer get(fn data_container): map hasher(blake2_128_concat) (T::AccountId,String) => Vec<String>;
	}
}

decl_event!(
	pub enum Event<T> where AccountId = <T as frame_system::Trait>::AccountId {
		MakeNewModel(NeuralKey<AccountId>,String),
		AddLayer(NeuralKey<AccountId>,String),
		UpdateModel(NeuralKey<AccountId>,String),
		AddDataSet(NeuralKey<AccountId>),
	}
);

// Errors inform users that something went wrong.
decl_error! {
	pub enum Error for Module<T: Trait> {
		UnableNewNueral,
		NoNueral,
		WrongLayerType 
	}
}

decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
		// Errors must be initialized if they are used by the pallet.
		type Error = Error<T>;

		// Events must be initialized if they are used by the pallet.
		fn deposit_event() = default;
		
		#[weight = 1_100_000_000]
		pub fn make_new_neural(origin, name: String) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;
			
			// make new model and 
			let ns = NeuralStruct::new(name.clone()); 
			Self::deposit_event(RawEvent::MakeNewModel((who.clone(),name.clone()),ns.get_model_string()));
			<NeuralContainer<T>>::insert((who.clone(),name.clone()), ns);
			Ok(())
		}
		
		#[weight = 2_500_000_000]
		pub fn add_layer(origin,name: String,size: (u32,u32),layer_type: String, extra_parameter: String) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;
			let (in_s,out_s) = (size.0 as usize, size.1 as usize);

			let mut ns = <NeuralContainer<T>>::get((who.clone(),name.clone()));


			let layer = match layer_type.as_str(){
				"HyperBolicTangent" =>{
					Some(NeuralLayer::new(in_s,out_s,HyperbolicTangent::new()))
				},
				"Sigmoid" =>{
					Some(NeuralLayer::new(in_s,out_s,Sigmoid::new()))
				},
				"LeackyLelu" =>{
					Some(NeuralLayer::new(in_s,out_s,LeakyRectifiedLinearUnit::new(extra_parameter.parse().unwrap())))
				},
				"SoftMax" =>{
					Some(NeuralLayer::new(in_s,out_s,SoftMax::new()))
				},
				"SoftPlus" =>{
					Some(NeuralLayer::new(in_s,out_s,SoftPlus::new()))
				},
				"Identity" =>{
					Some(NeuralLayer::new(in_s,out_s,Identity::new()))
				},
				_ =>{
					None
				}
			};
			ensure!(layer.is_some(),Error::<T>::WrongLayerType);
			ns.add_layers(layer.unwrap());
			Self::deposit_event(RawEvent::UpdateModel((who.clone(),name.clone()),ns.get_model_string()));
			<NeuralContainer<T>>::mutate((who.clone(),name.clone()),move |i|{
				*i=ns;
			});
			Ok(())
		}

		#[weight = 2_500_000_000]
		pub fn add_data_set(origin,name: String,size: (u32,u32), csv: String ) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoNueral);

			let mut reader = csv::Reader::from_reader(csv.as_bytes());
			let (in_s,out_s) = (size.0 as usize, size.1 as usize);

			let mut samples = Vec::new();
			for record in reader.records() {
				let record = record.unwrap();
				let result:Vec<f64> = record.iter().skip(in_s).take(out_s).map(|s| s.parse().unwrap()).collect();
				let input: Vec<f64> = record.iter().take(in_s).map(|s| s.parse().unwrap()).collect();
				let sample = Sample::new(input,result);
				samples.push(sample.to_string());
			}

			<DataContainer<T>>::insert((who.clone(),name.clone()), samples);
			
			Self::deposit_event(RawEvent::AddDataSet((who.clone(),name.clone())));
			Ok(())
		}
	}
}