#pragma once
#include "divGraph.h"

using dist_t = float;
using labeltype = int;
using tableint = int;

using namespace threadPoollib;

struct CompareByFirst {
	constexpr bool operator()(std::pair<dist_t, tableint> const& a,
		std::pair<dist_t, tableint> const& b) const noexcept {
		return a.first < b.first;
	}
};

class fastGraph
{
	std::string file;
	char* links = nullptr;
	size_t N = 0;
	size_t dim = 0;
	size_t maxT = 0;
	size_t size_data_per_element_;
	float** dataset = nullptr;
	//size_t max_elements_;
	const size_t sint = sizeof(int);
	threadPoollib::VisitedListPool* visited_list_pool_ = nullptr;
public:
	int ef = 0;
	int T = 0;
	int K = 0;
	int L = 0;

	std::string getFilename() const { return file; }

	fastGraph(divGraph* divG)
	{
		file = divG->getFilename();
		ef = divG->ef;
		N = divG->N;
		maxT = divG->maxT;
		size_data_per_element_ = (size_t)(maxT + 1) * sint;
		dataset = divG->myData;
		dim = divG->dim;
		visited_list_pool_ = new VisitedListPool(1, N);
		loadLite(divG);
	}

	void loadLite(divGraph* divG){
		links = (char*)malloc(N * size_data_per_element_);
		for (size_t i = 0; i < N; ++i) {
			char* begin = links + i * size_data_per_element_;
			auto& nns = divG->linkLists[i];
			memcpy(begin, &(nns->out), sint);
			begin += sint;
			for (int i = 0; i < nns->out; ++i) {
				memcpy(begin + i * sint, &(nns->neighbors[i].id), sint);
			}
		}
	}

	void knn(queryN* q){
		lsh::timer timer;
		timer.restart();

		std::priority_queue<std::pair<dist_t, labeltype >> result;
#ifdef USE_SSE
		_mm_prefetch((char*)(q->queryPoint), _MM_HINT_T0);
#endif
		
		int currObj = 0;
		int ep_id = 0;
		dist_t curdist = calL2Sqr_fast(q->queryPoint, dataset[ep_id], dim);
		q->cost++;
		VisitedList* vl = visited_list_pool_->getFreeVisitedList();
		vl_type* visited_array = vl->mass;
		vl_type visited_array_tag = vl->curV;

		std::priority_queue<std::pair<dist_t, tableint>> top_candidates;
		std::priority_queue<std::pair<dist_t, tableint>> candidate_set;
		//std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
		//std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

		dist_t lowerBound;
		dist_t dist = curdist;
		lowerBound = dist;
		top_candidates.emplace(dist, ep_id);
		candidate_set.emplace(-dist, ep_id);

		visited_array[ep_id] = visited_array_tag;

		while (!candidate_set.empty()) {

			std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

			if ((-current_node_pair.first) > lowerBound) {
				break;
			}
			candidate_set.pop();

			tableint current_node_id = current_node_pair.second;
			int* data = (int*)(links + current_node_id * size_data_per_element_);
			size_t size = *data;
			//bool cur_node_deleted = isMarkedDeleted(current_node_id);

#ifdef USE_SSE
			_mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
			_mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
			//_mm_prefetch(links + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
			_mm_prefetch((char*)(dataset[data[1]]), _MM_HINT_T0);
			_mm_prefetch((char*)(data + 1), _MM_HINT_T0);
#endif

			for (size_t j = 1; j <= size; j++) {
				int candidate_id = *(data + j);
				//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
				_mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
				_mm_prefetch((char*)(dataset[*(data + j + 1)]), _MM_HINT_T0);
#endif
				if (!(visited_array[candidate_id] == visited_array_tag)) {

					visited_array[candidate_id] = visited_array_tag;

					float* currObj1 = dataset[*(data + j)];
					dist_t dist = calL2Sqr_fast(q->queryPoint, currObj1, dim);
					q->cost++;
					if (top_candidates.size() < ef || lowerBound > dist) {
						candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
						_mm_prefetch((char*)(dataset[candidate_set.top().second]),_MM_HINT_T0);
#endif

						top_candidates.emplace(dist, candidate_id);
						if (top_candidates.size() > ef)
							top_candidates.pop();

						if (!top_candidates.empty())
							lowerBound = top_candidates.top().first;
					}
				}
			}
		}

		visited_list_pool_->releaseVisitedList(vl);

		while (top_candidates.size() > q->k) {
			top_candidates.pop();
		}
		q->res.resize(q->k);
		for (int i = q->k - 1; i > -1; --i) {
			std::pair<dist_t, tableint> rez = top_candidates.top();
			q->res[i] = Res(rez.first, rez.second);
			top_candidates.pop();
		}

		q->timeTotal = timer.elapsed();
		
	}

	void knnHNSW(queryN* q) {}
};