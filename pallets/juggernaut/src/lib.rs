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
use frame_support::sp_runtime::FixedI64;
use engine::sample::Sample;
use engine::matrix::Matrix;



#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;


pub trait Trait: frame_system::Trait {
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
		if self.neural_network.is_none(){
			self.neural_network=Some(NeuralNetwork::new().to_string());
		}
		if let Ok(mut net) = self.get_model(){
			net.add_layer(layer);
			self.neural_network=Some(net.to_string());
		}	
	}
	pub fn get_model_string(&self)->String{
		self.neural_network.as_ref().unwrap_or(&"".to_owned()).to_string()
	}
	pub fn train(&mut self,samples: Vec<Sample> ,epoch: i32, learning_rate: f64){
		let mut nn =self.get_model().unwrap();
		nn.train(samples,epoch,learning_rate,None);
		self.neural_network=Some(nn.to_string());
	}
	pub fn run(&self, samples: Sample) -> Matrix{
		let nn =self.get_model().unwrap();
		nn.evaluate(&samples)
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
		MakeNewModel(NeuralKey<AccountId>),
		AddLayer(NeuralKey<AccountId>,String),
		UpdateModel(NeuralKey<AccountId>,String),
		AddDataSet(NeuralKey<AccountId>),
		TrainComplete(NeuralKey<AccountId>),
		RunResult(NeuralKey<AccountId>,String),
	}
);

// Errors inform users that something went wrong.
decl_error! {
	pub enum Error for Module<T: Trait> {
		UnableNewNueral,
		NoModel,
		WrongLayerType,
		ModelParsingError,
		NoData
	}
}

decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
		// Errors must be initialized if they are used by the pallet.
		type Error = Error<T>;

		// Events must be initialized if they are used by the pallet.
		fn deposit_event() = default;
		
		// generate new model
		#[weight = 1_100_000_000]
		pub fn make_new_neural(origin, name: String) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;
			
			let ns = NeuralStruct::new(name.clone()); 
			Self::deposit_event(RawEvent::MakeNewModel((who.clone(),name.clone())));
			<NeuralContainer<T>>::insert((who.clone(),name.clone()), ns);
			Ok(())
		}
		
		//add layer to model
		#[weight = 2_500_000_000]
		pub fn add_layer(origin,name: String,size: (u32,u32),layer_type: String, extra_parameter: String) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;
			let (in_s,out_s) = (size.0 as usize, size.1 as usize);

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);

			let mut ns=<NeuralContainer<T>>::get((who.clone(),name.clone()));


			let layer = match layer_type.as_str(){
				"HyperBolicTangent" =>{
					Some(NeuralLayer::new(out_s,in_s,HyperbolicTangent::new()))
				},
				"Sigmoid" =>{
					Some(NeuralLayer::new(out_s,in_s,Sigmoid::new()))
				},
				"RectifiedLinear" =>{
					Some(NeuralLayer::new(out_s,in_s,RectifiedLinearUnit::new()))
				}
				"LeackyLelu" =>{
					Some(NeuralLayer::new(out_s,in_s,LeakyRectifiedLinearUnit::new(extra_parameter.parse().unwrap())))
				},
				"SoftMax" =>{
					Some(NeuralLayer::new(out_s,in_s,SoftMax::new()))
				},
				"SoftPlus" =>{
					Some(NeuralLayer::new(out_s,in_s,SoftPlus::new()))
				},
				"Identity" =>{
					Some(NeuralLayer::new(out_s,in_s,Identity::new()))
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

		//add data_set for model
		#[weight = 2_500_000_000]
		pub fn add_data_set(origin,name: String,size: (u32,u32), csv: String ) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);

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

		//train model
		#[weight = 2_500_000_000]
		pub fn train(origin,name: String,epoch: i32, learning_rate: FixedI64) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);

			let mut ns=<NeuralContainer<T>>::get((who.clone(),name.clone()));
			ensure!(ns.get_model().is_ok(),Error::<T>::ModelParsingError);

			ensure!(<DataContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoData);
			let datas = <DataContainer<T>>::get((who.clone(),name.clone())); 

			let mut samples = Vec::new();
			for raw_string in datas.iter(){
				match Sample::from_string(raw_string.clone()){
					Ok(s) => {
						samples.push(s)
					},
					Err(_) =>{
						
					}
				}
			}
			ns.train(samples,epoch,learning_rate.to_fraction());
			Self::deposit_event(RawEvent::TrainComplete((who.clone(),name.clone())));
			<NeuralContainer<T>>::mutate((who.clone(),name.clone()),move |i|{
				*i=ns
			});
			Ok(())
		}

		//run model
		#[weight = 2_500_000_000]
		pub fn run(origin,name: String, csv: String) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);
			let ns=<NeuralContainer<T>>::get((who.clone(),name.clone()));

			ensure!(ns.get_model().is_ok(),Error::<T>::ModelParsingError);
			let sample: Vec<f64>= csv.split(',').map(|s| s.parse().unwrap()).collect();
			let result=ns.run(Sample::predict(sample));
			
			Self::deposit_event(RawEvent::RunResult((who.clone(),name.clone()),result.to_string()));
			Ok(())
		}
	}
}