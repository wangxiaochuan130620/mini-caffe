#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>

#include "caffe/net.hpp"
#include "caffe/profiler.hpp"
#include "./layer.hpp"
#include "./util/math_functions.hpp"
#include "./util/upgrade_proto.hpp"
#include "./proto/caffe.pb.h"
#include "./util/insert_splits.hpp"
using namespace std;
namespace caffe {

	static bool StateMeetsRule(const NetState& state,
		const NetStateRule& rule, const std::string& layer_name) {
		// Check whether the rule is broken due to phase.
		if (rule.has_phase()) {
			if (rule.phase() != state.phase()) {
				LOG(INFO)
					<< "The NetState phase (" << state.phase()
					<< ") differed from the phase (" << rule.phase()
					<< ") specified by a rule in layer " << layer_name;
				return false;
			}
		}
		// Check whether the rule is broken due to min level.
		if (rule.has_min_level()) {
			if (state.level() < rule.min_level()) {
				LOG(INFO)
					<< "The NetState level (" << state.level()
					<< ") is above the min_level (" << rule.min_level()
					<< ") specified by a rule in layer " << layer_name;
				return false;
			}
		}
		// Check whether the rule is broken due to max level.
		if (rule.has_max_level()) {
			if (state.level() > rule.max_level()) {
				LOG(INFO)
					<< "The NetState level (" << state.level()
					<< ") is above the max_level (" << rule.max_level()
					<< ") specified by a rule in layer " << layer_name;
				return false;
			}
		}
		// Check whether the rule is broken due to stage. The NetState must
		// contain ALL of the rule's stages to meet it.
		for (int i = 0; i < rule.stage_size(); ++i) {
			// Check that the NetState contains the rule's ith stage.
			bool has_stage = false;
			for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
				if (rule.stage(i) == state.stage(j)) { has_stage = true; }
			}
			if (!has_stage) {
				LOG(INFO)
					<< "The NetState did not contain stage '" << rule.stage(i)
					<< "' specified by a rule in layer " << layer_name;
				return false;
			}
		}
		// Check whether the rule is broken due to not_stage. The NetState must
		// contain NONE of the rule's not_stages to meet it.
		for (int i = 0; i < rule.not_stage_size(); ++i) {
			// Check that the NetState contains the rule's ith not_stage.
			bool has_stage = false;
			for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
				if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
			}
			if (has_stage) {
				LOG(INFO)
					<< "The NetState contained a not_stage '" << rule.not_stage(i)
					<< "' specified by a rule in layer " << layer_name;
				return false;
			}
		}
		return true;
	}

	static void FilterNet(const NetParameter& param,
		NetParameter* param_filtered) {
		NetState net_state(param.state());
		param_filtered->CopyFrom(param);
		param_filtered->clear_layer();
		for (int i = 0; i < param.layer_size(); ++i) {
			const LayerParameter& layer_param = param.layer(i);
			const string& layer_name = layer_param.name();
			CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
				<< "Specify either include rules or exclude rules; not both.";
			// If no include rules are specified, the layer is included by default and
			// only excluded if it meets one of the exclude rules.
			bool layer_included = (layer_param.include_size() == 0);
			for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
				if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
					layer_included = false;
				}
			}
			for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
				if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
					layer_included = true;
				}
			}
			if (layer_included) {
				param_filtered->add_layer()->CopyFrom(layer_param);
			}
		}
	}

	Net::Net(const NetParameter& param)
	{
		Init(param);
	}

	 
	Net::Net(const string& param_file)
	{
		NetParameter param;
		ReadNetParamsFromTextFileOrDie(param_file, &param);
		Init(param);
	}

	 
	void Net::Init(const NetParameter& in_param) {
		// Filter layers based on their include/exclude rules and
		// the current NetState.
		NetParameter filtered_param;
		FilterNet(in_param, &filtered_param);
		// Create a copy of filtered_param with splits added where necessary.
		NetParameter param;
		InsertSplits(filtered_param, &param);
		// Basically, build all the layers and set up their connections.
		name_ = param.name();
		map<string, int> blob_name_to_idx;
		set<string> available_blobs;
		memory_used_ = 0;
		// For each layer, set up its input and output
		bottom_vecs_.resize(param.layer_size());
		top_vecs_.resize(param.layer_size());
		bottom_id_vecs_.resize(param.layer_size());
		param_id_vecs_.resize(param.layer_size());
		top_id_vecs_.resize(param.layer_size());
		bottom_need_backward_.resize(param.layer_size());
		for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
			// Setup layer.
			const LayerParameter& layer_param = param.layer(layer_id);
			layers_.push_back(LayerRegistry::CreateLayer(layer_param));
			layer_names_.push_back(layer_param.name());
			LOG(INFO) << "Creating Layer " << layer_param.name();
			bool need_backward = false;

			// Figure out this layer's input and output
			for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
				++bottom_id) {
				const int blob_id = AppendBottom(param, layer_id, bottom_id,
					&available_blobs, &blob_name_to_idx);
				// If a blob needs backward, this layer should provide it.
				need_backward |= blob_need_backward_[blob_id];
			}
			int num_top = layer_param.top_size();
			for (int top_id = 0; top_id < num_top; ++top_id) {
				AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
				// Collect Input layer tops as Net inputs.
				if (layer_param.type() == "Input") {
					const int blob_id = blobs_.size() - 1;
					net_input_blob_indices_.push_back(blob_id);
					net_input_blobs_.push_back(blobs_[blob_id].get());
				}
			}
			layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
			
			LOG(INFO) << "Setting up " << layer_names_[layer_id];
			for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
				memory_used_ += top_vecs_[layer_id][top_id]->count();
			}
			LOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(real_t);
			const int param_size = layer_param.param_size();
			const int num_param_blobs = layers_[layer_id]->blobs().size();
			CHECK_LE(param_size, num_param_blobs)
				<< "Too many params specified for layer " << layer_param.name();
			for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
				AppendParam(param, layer_id, param_id);
			}
		}
		
		// In the end, all remaining blobs are considered output blobs.
		for (set<string>::iterator it = available_blobs.begin();
			it != available_blobs.end(); ++it) {
			LOG(INFO) << "This network produces output " << *it;
			net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
			net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
		}
		for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
			blob_names_index_[blob_names_[blob_id]] = blob_id;
		}
		for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
			layer_names_index_[layer_names_[layer_id]] = layer_id;
		}
		LOG(INFO) << "Network initialization done.";
	}

	 
	// Helper for Net::Init: add a new top blob to the net.
	 
	void Net::AppendTop(const NetParameter& param, const int layer_id,
		const int top_id, set<string>* available_blobs,
		map<string, int>* blob_name_to_idx) {
		shared_ptr<LayerParameter> layer_param(
			new LayerParameter(param.layer(layer_id)));
		const string& blob_name = (layer_param->top_size() > top_id) ?
			layer_param->top(top_id) : "(automatic)";
		// Check if we are doing in-place computation
		if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
			blob_name == layer_param->bottom(top_id)) {
			// In-place computation
			LOG(INFO) << layer_param->name() << " -> " << blob_name << " (in-place)";
			top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
			top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
		}
		else if (blob_name_to_idx &&
			blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
			// If we are not doing in-place computation but have duplicated blobs,
			// raise an error.
			LOG(FATAL) << "Top blob '" << blob_name
				<< "' produced by multiple sources.";
		}
		else {
			// Normal output.
			shared_ptr<Blob > blob_pointer(new Blob());
			const int blob_id = blobs_.size();
			blobs_.push_back(blob_pointer);
			blob_names_.push_back(blob_name);
			blob_need_backward_.push_back(false);
			if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
			top_id_vecs_[layer_id].push_back(blob_id);
			top_vecs_[layer_id].push_back(blob_pointer.get());
		}
		if (available_blobs) { available_blobs->insert(blob_name); }
	}

	// Helper for Net::Init: add a new bottom blob to the net.
	 
	int Net::AppendBottom(const NetParameter& param, const int layer_id,
		const int bottom_id, set<string>* available_blobs,
		map<string, int>* blob_name_to_idx) {
		const LayerParameter& layer_param = param.layer(layer_id);
		const string& blob_name = layer_param.bottom(bottom_id);
		if (available_blobs->find(blob_name) == available_blobs->end()) {
			LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
				<< layer_param.name() << "', bottom index " << bottom_id << ")";
		}
		const int blob_id = (*blob_name_to_idx)[blob_name];
		LOG(INFO) << layer_names_[layer_id] << " <- " << blob_name;
		bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
		bottom_id_vecs_[layer_id].push_back(blob_id);
		available_blobs->erase(blob_name);
		bool need_backward = blob_need_backward_[blob_id];
		// Check if the backpropagation on bottom_id should be skipped
		if (layer_param.propagate_down_size() > 0) {
			need_backward = layer_param.propagate_down(bottom_id);
		}
		bottom_need_backward_[layer_id].push_back(need_backward);
		return blob_id;
	}

	 
	void Net::AppendParam(const NetParameter& param, const int layer_id,
		const int param_id) {
		const LayerParameter& layer_param = layers_[layer_id]->layer_param();
		const int param_size = layer_param.param_size();
		string param_name =
			(param_size > param_id) ? layer_param.param(param_id).name() : "";
		if (param_name.size()) {
			param_display_names_.push_back(param_name);
		}
		else {
			ostringstream param_display_name;
			param_display_name << param_id;
			param_display_names_.push_back(param_display_name.str());
		}
		const int net_param_id = params_.size();
		params_.push_back(layers_[layer_id]->blobs()[param_id]);
		param_id_vecs_[layer_id].push_back(net_param_id);
		param_layer_indices_.push_back(make_pair(layer_id, param_id));
		ParamSpec default_param_spec;
		const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
			&layer_param.param(param_id) : &default_param_spec;
		if (!param_size || !param_name.size() || (param_name.size() &&
			param_names_index_.find(param_name) == param_names_index_.end())) {
			// This layer "owns" this parameter blob -- it is either anonymous
			// (i.e., not given a param_name) or explicitly given a name that we
			// haven't already seen.
			param_owners_.push_back(-1);
			if (param_name.size()) {
				param_names_index_[param_name] = net_param_id;
			}
			const int learnable_param_id = learnable_params_.size();
			learnable_params_.push_back(params_[net_param_id].get());
			learnable_param_ids_.push_back(learnable_param_id);
			has_params_lr_.push_back(param_spec->has_lr_mult());
			has_params_decay_.push_back(param_spec->has_decay_mult());
			params_lr_.push_back(param_spec->lr_mult());
			params_weight_decay_.push_back(param_spec->decay_mult());
		}
		else {
			// Named param blob with name we've seen before: share params
			const int owner_net_param_id = param_names_index_[param_name];
			param_owners_.push_back(owner_net_param_id);
			const pair<int, int>& owner_index =
				param_layer_indices_[owner_net_param_id];
			const int owner_layer_id = owner_index.first;
			const int owner_param_id = owner_index.second;
			LOG(INFO) << "Sharing parameters '" << param_name
				<< "' owned by "
				<< "layer '" << layer_names_[owner_layer_id] << "', param "
				<< "index " << owner_param_id;
			Blob* this_blob = layers_[layer_id]->blobs()[param_id].get();
			Blob* owner_blob =
				layers_[owner_layer_id]->blobs()[owner_param_id].get();
			const int param_size = layer_param.param_size();
			if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
				ParamSpec_DimCheckMode_PERMISSIVE)) {
				// Permissive dimension checking -- only check counts are the same.
				CHECK_EQ(this_blob->count(), owner_blob->count())
					<< "Cannot share param '" << param_name << "' owned by layer '"
					<< layer_names_[owner_layer_id] << "' with layer '"
					<< layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
					<< "shape is " << owner_blob->shape_string() << "; sharing layer "
					<< "shape is " << this_blob->shape_string();
			}
			else {
				// Strict dimension checking -- all dims must be the same.
				CHECK(this_blob->shape() == owner_blob->shape())
					<< "Cannot share param '" << param_name << "' owned by layer '"
					<< layer_names_[owner_layer_id] << "' with layer '"
					<< layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
					<< "shape is " << owner_blob->shape_string() << "; sharing layer "
					<< "expects shape " << this_blob->shape_string();
			}
			const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
			learnable_param_ids_.push_back(learnable_param_id);
			if (param_spec->has_lr_mult()) {
				if (has_params_lr_[learnable_param_id]) {
					CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
						<< "Shared param '" << param_name << "' has mismatched lr_mult.";
				}
				else {
					has_params_lr_[learnable_param_id] = true;
					params_lr_[learnable_param_id] = param_spec->lr_mult();
				}
			}
			if (param_spec->has_decay_mult()) {
				if (has_params_decay_[learnable_param_id]) {
					CHECK_EQ(param_spec->decay_mult(),
						params_weight_decay_[learnable_param_id])
						<< "Shared param '" << param_name << "' has mismatched decay_mult.";
				}
				else {
					has_params_decay_[learnable_param_id] = true;
					params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
				}
			}
		}
	}

	 
	real_t Net::ForwardFromTo(int start, int end) {
		CHECK_GE(start, 0);
		CHECK_LT(end, layers_.size());
		real_t loss = 0;
		for (int i = start; i <= end; ++i) {
			// LOG(ERROR) << "Forwarding " << layer_names_[i];
			real_t layer_loss = 0;
			layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
			loss += layer_loss;
		}
		return loss;
	}

	const vector<Blob*>& Net::Forward(real_t* loss) {
		if (loss != NULL) {
			*loss = ForwardFromTo(0, layers_.size() - 1);
		}
		else {
			ForwardFromTo(0, layers_.size() - 1);
		}
		return net_output_blobs_;
	}

	 
	void Net::Reshape() {
		for (int i = 0; i < layers_.size(); ++i) {
			layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
		}
	}

	void Net::CopyTrainedLayersFrom(const NetParameter& param) {
		int num_source_layers = param.layer_size();
		for (int i = 0; i < num_source_layers; ++i) {
			const LayerParameter& source_layer = param.layer(i);
			const string& source_layer_name = source_layer.name();
			int target_layer_id = 0;
			while (target_layer_id != layer_names_.size() &&
				layer_names_[target_layer_id] != source_layer_name) {
				++target_layer_id;
			}
			if (target_layer_id == layer_names_.size()) {
				LOG(INFO) << "Ignoring source layer " << source_layer_name;
				continue;
			}
			DLOG(INFO) << "Copying source layer " << source_layer_name;
			vector<shared_ptr<Blob > >& target_blobs =
				layers_[target_layer_id]->blobs();
			CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
				<< "Incompatible number of blobs for layer " << source_layer_name;
			for (int j = 0; j < target_blobs.size(); ++j) {
				if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
					Blob source_blob;
					const bool kReshape = true;
					source_blob.FromProto(source_layer.blobs(j), kReshape);
					LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
						<< source_layer_name << "'; shape mismatch.  Source param shape is "
						<< source_blob.shape_string() << "; target param shape is "
						<< target_blobs[j]->shape_string() << ". "
						<< "To learn this layer's parameters from scratch rather than "
						<< "copying from a saved net, rename the layer.";
				}
				const bool kReshape = false;
				target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
			}
		}
	}

	void Net::CopyTrainedLayersFrom(const string trained_filename) {
		CopyTrainedLayersFromBinaryProto(trained_filename);
	}

	 
	void Net::CopyTrainedLayersFromBinaryProto(
		const string trained_filename) {
		NetParameter param;
		ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
		CopyTrainedLayersFrom(param);
	}

	 
	bool Net::has_blob(const string& blob_name) const {
		return blob_names_index_.find(blob_name) != blob_names_index_.end();
	}

	 
	const shared_ptr<Blob > Net::blob_by_name(
		const string& blob_name) const {
		shared_ptr<Blob > blob_ptr;
		if (has_blob(blob_name)) {
			blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
		}
		else {
			blob_ptr.reset((Blob*)(NULL));
			LOG(WARNING) << "Unknown blob name " << blob_name;
		}
		return blob_ptr;
	}

	bool Net::has_layer(const string& layer_name) const {
		return layer_names_index_.find(layer_name) != layer_names_index_.end();
	}

	const shared_ptr<Layer > Net::layer_by_name(
		const string& layer_name) const {
		shared_ptr<Layer > layer_ptr;
		if (has_layer(layer_name)) {
			layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
		}
		else {
			layer_ptr.reset((Layer*)(NULL));
			LOG(WARNING) << "Unknown layer name " << layer_name;
		}
		return layer_ptr;
	}

//////////////////////


}  // namespace caffe
