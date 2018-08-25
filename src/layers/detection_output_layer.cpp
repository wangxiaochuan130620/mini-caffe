#include <cfloat>

#include "./detection_output_layer.hpp"
#include "../util/math_functions.hpp"

using std::map;
using std::pair;

namespace caffe {

void DetectionOutputLayer::LayerSetUp(const vector<Blob*>& bottom,
                                      const vector<Blob*>& top) {
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();
  CHECK(detection_output_param.has_num_classes()) << "Must specify num_classes";
  num_classes_ = detection_output_param.num_classes();
  share_location_ = detection_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;
  background_label_id_ = detection_output_param.background_label_id();
  code_type_ = detection_output_param.code_type();
  variance_encoded_in_target_ =
      detection_output_param.variance_encoded_in_target();
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -FLT_MAX;
  // Parameters used in nms.
  nms_threshold_ = detection_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
  eta_ = detection_output_param.nms_param().eta();
  CHECK_GT(eta_, 0.);
  CHECK_LE(eta_, 1.);
  top_k_ = -1;
  if (detection_output_param.nms_param().has_top_k()) {
    top_k_ = detection_output_param.nms_param().top_k();
  }
  bbox_preds_.ReshapeLike(*(bottom[0]));
  if (!share_location_) {
    bbox_permute_.ReshapeLike(*(bottom[0]));
  }
  conf_permute_.ReshapeLike(*(bottom[1]));
}

void DetectionOutputLayer::Reshape(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top) {
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	num_priors_ = bottom[2]->height() / 4;
	CHECK_EQ(num_priors_ * num_loc_classes_ * 4, bottom[0]->channels())
		<< "Number of priors must match number of location predictions.";
	CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
		<< "Number of priors must match number of confidence predictions.";
	// num() and channels() are 1.
	vector<int> top_shape(2, 1);
	// Since the number of bboxes to be kept is unknown before nms, we manually
	// set it to (fake) 1.
	top_shape.push_back(1);
	// Each row is a 7 dimension vector, which stores
	// [image_id, label, confidence, xmin, ymin, xmax, ymax]
	top_shape.push_back(7);
	top[0]->Reshape(top_shape);
}


void GetMaxScoreIndexEx(const vector<float>& scores, const float threshold,
	const int top_k, vector<pair<float, int> >* score_index_vec) {
	// Generate index score pairs.
	for (int i = 0; i < scores.size(); ++i) {
		if (scores[i] > threshold) {
			score_index_vec->push_back(std::make_pair(scores[i], i));
		}
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
		SortScorePairDescend<int>);

	// Keep top_k scores if needed.
	if (top_k > -1 && top_k < score_index_vec->size()) {
		score_index_vec->resize(top_k);
	}
}

void ApplyNMSFastEx(const vector<NormalizedBBox>& bboxes,
	const vector<float>& scores, const float score_threshold,
	const float nms_threshold, const int top_k, vector<int>* indices) {
	// Sanity check.
	CHECK_EQ(bboxes.size(), scores.size())
		<< "bboxes and scores have different size.";

	// Get top_k scores (with corresponding indices).
	vector<pair<float, int> > score_index_vec;
	GetMaxScoreIndexEx(scores, score_threshold, top_k, &score_index_vec);

	// Do nms.
	indices->clear();
	while (score_index_vec.size() != 0) {
		const int idx = score_index_vec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); ++k) {
			if (keep) {
				const int kept_idx = (*indices)[k];
				float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
				keep = overlap <= nms_threshold;
			}
			else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		score_index_vec.erase(score_index_vec.begin());
	}
}


void DecodeBBoxEx(
	const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
	const CodeType code_type, const bool variance_encoded_in_target,
	const NormalizedBBox& bbox, NormalizedBBox* decode_bbox) {
	if (code_type == PriorBoxParameter_CodeType_CORNER) {
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to add the offset
			// predictions.
			decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
			decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
			decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
			decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
		}
		else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decode_bbox->set_xmin(
				prior_bbox.xmin() + prior_variance[0] * bbox.xmin());
			decode_bbox->set_ymin(
				prior_bbox.ymin() + prior_variance[1] * bbox.ymin());
			decode_bbox->set_xmax(
				prior_bbox.xmax() + prior_variance[2] * bbox.xmax());
			decode_bbox->set_ymax(
				prior_bbox.ymax() + prior_variance[3] * bbox.ymax());
		}
	}
	else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
		float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
		CHECK_GT(prior_width, 0);
		float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
		CHECK_GT(prior_height, 0);
		float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
		float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

		float decode_bbox_center_x, decode_bbox_center_y;
		float decode_bbox_width, decode_bbox_height;
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to retore the offset
			// predictions.
			decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
			decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
			decode_bbox_width = exp(bbox.xmax()) * prior_width;
			decode_bbox_height = exp(bbox.ymax()) * prior_height;
		}
		else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decode_bbox_center_x =
				prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
			decode_bbox_center_y =
				prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
			decode_bbox_width =
				exp(prior_variance[2] * bbox.xmax()) * prior_width;
			decode_bbox_height =
				exp(prior_variance[3] * bbox.ymax()) * prior_height;
		}

		decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
		decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
		decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
		decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
	}
	else {
		LOG(FATAL) << "Unknown LocLossType.";
	}
	float bbox_size = BBoxSize(*decode_bbox);
	decode_bbox->set_size(bbox_size);
}

void DecodeBBoxesEx(
	const vector<NormalizedBBox>& prior_bboxes,
	const vector<vector<float> >& prior_variances,
	const CodeType code_type, const bool variance_encoded_in_target,
	const vector<NormalizedBBox>& bboxes,
	vector<NormalizedBBox>* decode_bboxes) {
	CHECK_EQ(prior_bboxes.size(), prior_variances.size());
	CHECK_EQ(prior_bboxes.size(), bboxes.size());
	int num_bboxes = prior_bboxes.size();
	if (num_bboxes >= 1) {
		CHECK_EQ(prior_variances[0].size(), 4);
	}
	decode_bboxes->clear();
	for (int i = 0; i < num_bboxes; ++i) {
		NormalizedBBox decode_bbox;
		DecodeBBoxEx(prior_bboxes[i], prior_variances[i], code_type,
			variance_encoded_in_target, bboxes[i], &decode_bbox);
		decode_bboxes->push_back(decode_bbox);
	}
}

void DecodeBBoxesAllEx(const vector<LabelBBox>& all_loc_preds,
	const vector<NormalizedBBox>& prior_bboxes,
	const vector<vector<float> >& prior_variances,
	const int num, const bool share_location,
	const int num_loc_classes, const int background_label_id,
	const CodeType code_type, const bool variance_encoded_in_target,
	vector<LabelBBox>* all_decode_bboxes) {
	CHECK_EQ(all_loc_preds.size(), num);
	all_decode_bboxes->clear();
	all_decode_bboxes->resize(num);
	for (int i = 0; i < num; ++i) {
		// Decode predictions into bboxes.
		LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
		for (int c = 0; c < num_loc_classes; ++c) {
			int label = share_location ? -1 : c;
			if (label == background_label_id) {
				// Ignore background class.
				continue;
			}
			if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for label " << label;
			}
			const vector<NormalizedBBox>& label_loc_preds =
				all_loc_preds[i].find(label)->second;
			DecodeBBoxesEx(prior_bboxes, prior_variances,
				code_type, variance_encoded_in_target,
				label_loc_preds, &(decode_bboxes[label]));
		}
	}
}

void DetectionOutputLayer::Forward_cpu(const vector<Blob*>& bottom,
                                       const vector<Blob*>& top) {
  const real_t* loc_data = bottom[0]->cpu_data();
  const real_t* conf_data = bottom[1]->cpu_data();
  const real_t* prior_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num();

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
	  share_location_, &all_loc_preds);

  // Retrieve all confidences.
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
	  &all_conf_scores);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Decode all loc predictions to bboxes.
  vector<LabelBBox> all_decode_bboxes;
  DecodeBBoxesAllEx(all_loc_preds, prior_bboxes, prior_variances, num,
	  share_location_, num_loc_classes_, background_label_id_,
	  code_type_, variance_encoded_in_target_, &all_decode_bboxes);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
	  const LabelBBox& decode_bboxes = all_decode_bboxes[i];
	  const map<int, vector<float> >& conf_scores = all_conf_scores[i];
	  map<int, vector<int> > indices;
	  int num_det = 0;
	  for (int c = 0; c < num_classes_; ++c) {
		  if (c == background_label_id_) {
			  // Ignore background class.
			  continue;
		  }
		  if (conf_scores.find(c) == conf_scores.end()) {
			  // Something bad happened if there are no predictions for current label.
			  LOG(FATAL) << "Could not find confidence predictions for label " << c;
		  }
		  const vector<float>& scores = conf_scores.find(c)->second;
		  int label = share_location_ ? -1 : c;
		  if (decode_bboxes.find(label) == decode_bboxes.end()) {
			  // Something bad happened if there are no predictions for current label.
			  LOG(FATAL) << "Could not find location predictions for label " << label;
			  continue;
		  }
		  const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
		  ApplyNMSFastEx(bboxes, scores, confidence_threshold_, nms_threshold_,
			  top_k_, &(indices[c]));
		  num_det += indices[c].size();
	  }
	  if (keep_top_k_ > -1 && num_det > keep_top_k_) {
		  vector<pair<float, pair<int, int> > > score_index_pairs;
		  for (map<int, vector<int> >::iterator it = indices.begin();
			  it != indices.end(); ++it) {
			  int label = it->first;
			  const vector<int>& label_indices = it->second;
			  if (conf_scores.find(label) == conf_scores.end()) {
				  // Something bad happened for current label.
				  LOG(FATAL) << "Could not find location predictions for " << label;
				  continue;
			  }
			  const vector<float>& scores = conf_scores.find(label)->second;
			  for (int j = 0; j < label_indices.size(); ++j) {
				  int idx = label_indices[j];
				  CHECK_LT(idx, scores.size());
				  score_index_pairs.push_back(std::make_pair(
					  scores[idx], std::make_pair(label, idx)));
			  }
		  }
		  // Keep top k results per image.
		  std::sort(score_index_pairs.begin(), score_index_pairs.end(),
			  SortScorePairDescend<pair<int, int> >);
		  score_index_pairs.resize(keep_top_k_);
		  // Store the new indices.
		  map<int, vector<int> > new_indices;
		  for (int j = 0; j < score_index_pairs.size(); ++j) {
			  int label = score_index_pairs[j].second.first;
			  int idx = score_index_pairs[j].second.second;
			  new_indices[label].push_back(idx);
		  }
		  all_indices.push_back(new_indices);
		  num_kept += keep_top_k_;
	  }
	  else {
		  all_indices.push_back(indices);
		  num_kept += num_det;
	  }
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);
  if (num_kept == 0) {
	  LOG(INFO) << "Couldn't find any detections";
	  top_shape[2] = 1;
	  top[0]->Reshape(top_shape);
	  caffe_set(top[0]->count(), -1, top[0]->mutable_cpu_data());
	  return;
  }
  top[0]->Reshape(top_shape);
  real_t* top_data = top[0]->mutable_cpu_data();

  int count = 0;
  for (int i = 0; i < num; ++i) {
	  const map<int, vector<float> >& conf_scores = all_conf_scores[i];
	  const LabelBBox& decode_bboxes = all_decode_bboxes[i];
	  for (map<int, vector<int> >::iterator it = all_indices[i].begin();
		  it != all_indices[i].end(); ++it) {
		  int label = it->first;
		  if (conf_scores.find(label) == conf_scores.end()) {
			  // Something bad happened if there are no predictions for current label.
			  LOG(FATAL) << "Could not find confidence predictions for " << label;
			  continue;
		  }
		  const vector<float>& scores = conf_scores.find(label)->second;
		  int loc_label = share_location_ ? -1 : label;
		  if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
			  // Something bad happened if there are no predictions for current label.
			  LOG(FATAL) << "Could not find location predictions for " << loc_label;
			  continue;
		  }
		  const vector<NormalizedBBox>& bboxes =
			  decode_bboxes.find(loc_label)->second;
		  vector<int>& indices = it->second;
		  for (int j = 0; j < indices.size(); ++j) {
			  int idx = indices[j];
			  top_data[count * 7] = i;
			  top_data[count * 7 + 1] = label;
			  top_data[count * 7 + 2] = scores[idx];
			  NormalizedBBox clip_bbox;
			  ClipBBox(bboxes[idx], &clip_bbox);
			  top_data[count * 7 + 3] = clip_bbox.xmin();
			  top_data[count * 7 + 4] = clip_bbox.ymin();
			  top_data[count * 7 + 5] = clip_bbox.xmax();
			  top_data[count * 7 + 6] = clip_bbox.ymax();
			  ++count;
		  }
	  }
  }
}

#ifndef USE_CUDA
STUB_GPU(DetectionOutputLayer);
#endif

REGISTER_LAYER_CLASS(DetectionOutput);

}  // namespace caffe
