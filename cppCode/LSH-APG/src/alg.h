#pragma once
#include <iostream>
#include <fstream>
#include "Preprocess.h"
#include "divGraph.h"
#include "fastGraph.h"
#include "Query.h"
#include <time.h>
#include "basis.h"
#include <sstream>

#if defined(unix) || defined(__unix__)
struct llt
{
	int date, h, m, s;
	llt(size_t diff) { set(diff); }
	void set(size_t diff)
	{
		date = diff / 86400;
		diff = diff % 86400;
		h = diff / 3600;
		diff = diff % 3600;
		m = diff / 60;
		s = diff % 60;
	}
};
#endif

template <class Graph>
void graphSearch(float c, int k, Graph* myGraph, Preprocess& prep, float beta, std::string& datasetName, std::string& data_fold, int qType) {
	if (!myGraph) return;

	lsh::timer timer;
	int Qnum = 100;
	Performance perform;
	

	for (unsigned j = 0; j < Qnum; j++)
	{
		queryN* q = new queryN(j, c, k, prep, beta);
		switch (qType % 2) {
		case 0:
			myGraph->knn(q);
			break;
		case 1:
			myGraph->knnHNSW(q);
			break;
		}
		perform.update(q, prep);
	}

	
	std::string algName, qt;
	switch (qType/2) {
	case 0:
		algName = "fastG";
		break;
	case 1:
		algName = "divGraph";
		break;
	}

	switch (qType % 2) {
	case 0:
		qt = "Fast";
		break;
	case 1:
		qt = "HNSW";
		break;
	}
	float mean_time = (float)perform.timeTotal / perform.num;
	float cost = ((float)perform.cost) / ((float)perform.num);
	float ratio = ((float)perform.prunings) / (perform.cost);
	float cost_total = myGraph->S + cost / (1 - ratio) * (((float)myGraph->lowDim) / myGraph->dim) + cost;
	float cpq = myGraph->L * myGraph->K+_lsh_UB+cost / (1 - ratio) * (((float)myGraph->lowDim) / myGraph->dim);

	std::stringstream ss;
	ss << std::setw(_lspace) << algName
		<< std::setw(_sspace) << k
		<< std::setw(_sspace) << myGraph->ef
		<< std::setw(_lspace) << mean_time * 1000
		<< std::setw(_lspace) << ((float)perform.NN_num) / (perform.num * k)
		//<< std::setw(_lspace) << ((float)perform.ratio) / (perform.resNum)
		<< std::setw(_lspace) << ((float)perform.cost) / ((float)perform.num)// * prep.data.N)
		<< std::setw(_lspace) << cpq
		<< std::setw(_lspace) << cost_total
		<< std::setw(_lspace) << ((float)perform.prunings) / (perform.cost)
		//<< std::setw(_lspace) << ((float)perform.maxHop) / (perform.num)
		<< std::endl;

	time_t now = time(0);
	
	time_t zero_point = 1635153971;//Let me set the time at 2021.10.25. 17:27 as the zero point
	size_t diff = (size_t)(now - zero_point);

	std::cout << ss.str();

	std::string query_result(myGraph->getFilename());
	auto idx = query_result.rfind('/');
	query_result.assign(query_result.begin(), query_result.begin() + idx + 1);
	query_result += "result.txt";
	std::ofstream osf(query_result, std::ios_base::app);
	osf.seekp(0, std::ios_base::end);
	osf << ss.str();
	osf.close();

	
	float date = ((float)(now - zero_point)) / 86400;

	std::string fpath = data_fold + "ANN/";

	if (!GenericTool::CheckPathExistence(fpath.c_str())) {
		GenericTool::EnsurePathExistence(fpath.c_str());
		std::cout << BOLDGREEN << "WARNING:\n" << GREEN << "Could not find the path of result file. Have created it. \n"
			<< "The query result will be stored in: " << fpath.c_str() << RESET;
	}
	std::ofstream os(fpath + "LSH-G_div_result.csv", std::ios_base::app);
	if (os) {
		os.seekp(0, std::ios_base::end); // move to the end of file
		int tmp = (int)os.tellp();
		if (tmp == 0) {
			os << "Dataset,k,L,K,T,RATIO,RECALL,AVG_TIME,COST,DATE" << std::endl;
		}
		std::string dataset = datasetName;
		os << dataset << ',' << k << ',' << myGraph->L << ',' << myGraph->K << ',' 
			<< myGraph->T << ','
			<< ((float)perform.ratio) / (perform.resNum) << ','
			<< ((float)perform.NN_num) / (perform.num * k) << ','
			<< mean_time * 1000 << ','
			<< ((float)perform.cost) / (perform.num * prep.data.N) << ','
			<< date << ','
			<< std::endl;
		os.close();
	}
}

void zlshKnn(float c, int k, e2lsh& myLsh, Preprocess& prep, float beta, std::string& datasetName, std::string& data_fold) {

	int T = 10;

	Parameter param(prep, 10, 10, 1.0f);
	param.W = 0.3f;
	zlsh myZlsh(prep, param, "");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING ZQUERY ..." << std::endl;
	int Qnum = 100;
	lsh::progress_display pd(Qnum);
	Performance perform;
	for (unsigned j = 0; j < Qnum; j++)
	{
		queryN* q = new queryN(j, c, k, prep, beta);
		//myZlsh.knn(q);
		myZlsh.knnBestFirst(q);
		perform.update(q, prep);
		++pd;
	}

	myZlsh.testLLCP();
	//exit(0);

	showMemoryInfo();

	//for (int i = 0; i < perform.costs.size(); ++i) {
	//	printf("AVG COST(Table %d) : %f\n", i, ((float)perform.costs[i]) / ((float)perform.cost));
	//}

	float mean_time = (float)perform.timeTotal / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.resNum) << std::endl;
	std::cout << "AVG COST:          " << ((float)perform.cost) / ((float)perform.num * prep.data.N) << std::endl;
	std::cout << "\nQUERY FINISH... \n\n\n";

	time_t now = std::time(0);
	time_t zero_point = 1635153971;//Let me set the time at 2021.10.25. 17:27 as the zero point
	float date = ((float)(now - zero_point)) / 86400;

	std::string fpath = data_fold + "ANN/";

	if (!GenericTool::CheckPathExistence(fpath.c_str())) {
		GenericTool::EnsurePathExistence(fpath.c_str());
		std::cout << BOLDGREEN << "WARNING:\n" << GREEN << "Could not find the path of result file. Have created it. \n"
			<< "The query result will be stored in: " << fpath.c_str() << RESET;
	}
	std::ofstream os(fpath + "ZLSH_result.csv", std::ios_base::app);
	if (os) {
		os.seekp(0, std::ios_base::end); // move to the end of file
		int tmp = (int)os.tellp();
		if (tmp == 0) {
			os << "Dataset,c,k,L,K,RATIO,RECALL,AVG_TIME,COST,DATE" << std::endl;
		}
		std::string dataset = datasetName;
		os << dataset << ',' << c << ',' << k << ',' << myLsh.L << ',' << myLsh.K << ','
			<< ((float)perform.ratio) / (perform.resNum) << ','
			<< ((float)perform.NN_num) / (perform.num * k) << ','
			<< mean_time * 1000 << ','
			<< ((float)perform.cost) / (perform.num * prep.data.N) << ','
			<< date << ','
			<< std::endl;
		os.close();
	}
}

bool find_file(std::string&& file)
{
	std::ifstream in(file);
	return in.good();
}