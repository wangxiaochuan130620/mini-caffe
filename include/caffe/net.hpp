#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"

namespace caffe {

class Layer;
class NetParameter;

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
class CAFFE_API Net {
public:
	explicit Net(const NetParameter& param);
	explicit Net(const string& param_file);
	virtual ~Net() {}

	/// @brief Initialize a network with a NetParameter.
	void Init(const NetParameter& param);

	/**
	* @brief Run Forward and return the result.
	*
	*/
	const vector<Blob*>& Forward(real_t* loss = NULL);

	/**
	* The From and To variants of Forward and Backward operate on the
	* (topological) ordering by which the net is specified. For general DAG
	* networks, note that (1) computing from one layer to another might entail
	* extra computation on unrelated branches, and (2) computation starting in
	* the middle may be incorrect if all of the layers of a fan-in are not
	* included.
	*/
	real_t ForwardFromTo(int start, int end);

	/**
	* @brief Reshape all layers from bottom to top.
	*
	* This is useful to propagate changes to layer sizes without running
	* a forward pass, e.g. to compute output feature size.
	*/
	void Reshape();

	
	void CopyTrainedLayersFrom(const string trained_filename);
	void CopyTrainedLayersFromBinaryProto(const string trained_filename);
	void CopyTrainedLayersFrom(const NetParameter& param);

	/// @brief returns the network name.
	inline const string& name() const { return name_; }
	/// @brief returns the layer names
	inline const vector<string>& layer_names() const { return layer_names_; }
	/// @brief returns the blob names
	inline const vector<string>& blob_names() const { return blob_names_; }
	/// @brief returns the blobs
	inline const vector<shared_ptr<Blob > >& blobs() const {
		return blobs_;
	}
	/// @brief returns the layers
	inline const vector<shared_ptr<Layer > >& layers() const {
		return layers_;
	}

	/**
	* @brief returns the bottom vecs for each layer -- usually you won't
	*        need this unless you do per-layer checks such as gradients.
	*/
	inline const vector<vector<Blob*> >& bottom_vecs() const {
		return bottom_vecs_;
	}
	/**
	* @brief returns the top vecs for each layer -- usually you won't
	*        need this unless you do per-layer checks such as gradients.
	*/
	inline const vector<vector<Blob*> >& top_vecs() const {
		return top_vecs_;
	}
	/// @brief returns the ids of the top blobs of layer i
	inline const vector<int> & top_ids(int i) const {
		CHECK_GE(i, 0) << "Invalid layer id";
		CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
		return top_id_vecs_[i];
	}
	/// @brief returns the ids of the bottom blobs of layer i
	inline const vector<int> & bottom_ids(int i) const {
		CHECK_GE(i, 0) << "Invalid layer id";
		CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
		return bottom_id_vecs_[i];
	}
	inline const vector<vector<bool> >& bottom_need_backward() const {
		return bottom_need_backward_;
	}
	inline const vector<real_t>& blob_loss_weights() const {
		return blob_loss_weights_;
	}
	inline const vector<bool>& layer_need_backward() const {
		return layer_need_backward_;
	}
	/// @brief returns the parameters
	inline const vector<shared_ptr<Blob > >& params() const {
		return params_;
	}
	inline const vector<Blob*>& learnable_params() const {
		return learnable_params_;
	}
	/// @brief returns the learnable parameter learning rate multipliers
	inline const vector<float>& params_lr() const { return params_lr_; }
	inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
	/// @brief returns the learnable parameter decay multipliers
	inline const vector<float>& params_weight_decay() const {
		return params_weight_decay_;
	}
	inline const vector<bool>& has_params_decay() const {
		return has_params_decay_;
	}
	const std::map<string, int>& param_names_index() const {
		return param_names_index_;
	}
	inline const vector<int>& param_owners() const { return param_owners_; }
	inline const vector<string>& param_display_names() const {
		return param_display_names_;
	}
	/// @brief Input and output blob numbers
	inline int num_inputs() const { return net_input_blobs_.size(); }
	inline int num_outputs() const { return net_output_blobs_.size(); }
	inline const vector<Blob*>& input_blobs() const {
		return net_input_blobs_;
	}
	inline const vector<Blob*>& output_blobs() const {
		return net_output_blobs_;
	}
	inline const vector<int>& input_blob_indices() const {
		return net_input_blob_indices_;
	}
	inline const vector<int>& output_blob_indices() const {
		return net_output_blob_indices_;
	}
	bool has_blob(const string& blob_name) const;
	const shared_ptr<Blob > blob_by_name(const string& blob_name) const;
	bool has_layer(const string& layer_name) const;
	const shared_ptr<Layer > layer_by_name(const string& layer_name) const;
	const vector<string>& param_names() const { return param_display_names_; }
protected:
	// Helpers for Init.
	/// @brief Append a new top blob to the net.
	void AppendTop(const NetParameter& param, const int layer_id,
		const int top_id, std::set<string>* available_blobs,
		std::map<string, int>* blob_name_to_idx);
	/// @brief Append a new bottom blob to the net.
	int AppendBottom(const NetParameter& param, const int layer_id,
		const int bottom_id, std::set<string>* available_blobs,
		std::map<string, int>* blob_name_to_idx);
	/// @brief Append a new parameter blob to the net.
	void AppendParam(const NetParameter& param, const int layer_id,
		const int param_id);

	/// @brief The network name
	string name_;
	/// @brief The phase: TRAIN or TEST
	/// @brief Individual layers in the net
	vector<shared_ptr<Layer > > layers_;
	vector<string> layer_names_;
	std::map<string, int> layer_names_index_;
	vector<bool> layer_need_backward_;
	/// @brief the blobs storing intermediate results between the layer.
	vector<shared_ptr<Blob > > blobs_;
	vector<string> blob_names_;
	std::map<string, int> blob_names_index_;
	vector<bool> blob_need_backward_;
	/// bottom_vecs stores the vectors containing the input for each layer.
	/// They don't actually host the blobs (blobs_ does), so we simply store
	/// pointers.
	vector<vector<Blob*> > bottom_vecs_;
	vector<vector<int> > bottom_id_vecs_;
	vector<vector<bool> > bottom_need_backward_;
	/// top_vecs stores the vectors containing the output for each layer
	vector<vector<Blob*> > top_vecs_;
	vector<vector<int> > top_id_vecs_;
	/// Vector of weight in the loss (or objective) function of each net blob,
	/// indexed by blob_id.
	vector<real_t> blob_loss_weights_;
	vector<vector<int> > param_id_vecs_;
	vector<int> param_owners_;
	vector<string> param_display_names_;
	vector<std::pair<int, int> > param_layer_indices_;
	std::map<string, int> param_names_index_;
	/// blob indices for the input and the output of the net
	vector<int> net_input_blob_indices_;
	vector<int> net_output_blob_indices_;
	vector<Blob*> net_input_blobs_;
	vector<Blob*> net_output_blobs_;
	/// The parameters in the network.
	vector<shared_ptr<Blob > > params_;
	vector<Blob*> learnable_params_;
	/**
	* The mapping from params_ -> learnable_params_: we have
	* learnable_param_ids_.size() == params_.size(),
	* and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
	* if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
	* and learnable_params_[learnable_param_ids_[i]] gives its owner.
	*/
	vector<int> learnable_param_ids_;
	/// the learning rate multipliers for learnable_params_
	vector<float> params_lr_;
	vector<bool> has_params_lr_;
	/// the weight decay multipliers for learnable_params_
	vector<float> params_weight_decay_;
	vector<bool> has_params_decay_;
	/// The bytes of memory used by this net
	size_t memory_used_;
	/// Whether to compute and display debug info for the net.
	bool debug_info_;
	/// The root net that actually holds the shared layers in data parallelism
	DISABLE_COPY_AND_ASSIGN(Net);
};
}  // namespace caffe

#endif  // CAFFE_NET_HPP_
