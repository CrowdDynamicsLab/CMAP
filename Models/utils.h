#ifndef utils_H
#define utils_H

#include "CMAP.h"

using namespace std;

unsigned seed = chrono::system_clock::now().time_since_epoch().count();
mt19937 gen(seed);

void make_dir(string dir_path)
{
	struct stat st = {0};

	if (stat(dir_path.c_str(), &st) == -1) {
		mkdir(dir_path.c_str(), 0700);
	}
}

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


double mean(vector<double> v)
{
	double sum=0;
	for(int i=0;i<v.size();i++) sum+=v[i];

	return (1.0*sum)/v.size();
}

double standard_deviation(vector<double> v, double ave)
{

    double E=0;

    double inverse = 1.0 / static_cast<double>(v.size());
    for(int i=0; i<v.size(); i++)
    {
        E += pow( (v[i] - ave), 2);
    }
    return sqrt(inverse * E);
}


void quicksort(vector<pair<int, double> > & vect, int left, int right) {
	int l_hold, r_hold;
	pair<int, double> pivot;

	l_hold = left;
	r_hold = right;    
	int pivotidx = left;
	pivot = vect[pivotidx];

	while (left < right) {
		while (vect[right].second <= pivot.second && left < right) {
			right--;
		}
		if (left != right) {
			vect[left] = vect[right];
			left++;
		}
		while (vect[left].second >= pivot.second && left < right) {
			left++;
		}
		if (left != right) {
			vect[right] = vect[left];
			right--;
		}
	}

	vect[left] = pivot;
	pivotidx = left;
	left = l_hold;
	right = r_hold;

	if (left < pivotidx) {
		quicksort(vect, left, pivotidx - 1);
	}
	if (right > pivotidx) {
		quicksort(vect, pivotidx + 1, right);
	}    
}


int sample_using_norm(vector<double> probs)
{
	srand(time(0));

	int sampled_idx;
	int num = probs.size();

	// cummulate multinomial parameters
	for (int j = 1; j < num; j++) {
		probs[j] += probs[j - 1];
	}

	// scaled sample because of unnormalized p[]
	double u = ((double)rand() / RAND_MAX) * probs[num - 1];

	for(sampled_idx = 0; sampled_idx < num; sampled_idx++) {
		if (probs[sampled_idx] > u) {
			break;
		}
	}

	if(sampled_idx == num) sampled_idx = num-1;

	return sampled_idx;
}


int sample_from_prob(vector<double> probs)
{
	discrete_distribution<> d(probs.begin(), probs.end());

	int sampled_idx = d(gen);

	return sampled_idx;
}

vector<double> handle_underflow(vector<double> probs)
{
	int num = probs.size();

	double max_prob = *max_element(probs.begin(), probs.begin()+num);

	for(int j = 0; j < num; ++j)
	{
		
		if(probs[j] > 1) probs[j] = 0.0;
		else
		{
			probs[j] = probs[j]-max_prob;

			if(probs[j] < MIN_LOG)
			{
				probs[j] = 0.0;
			}

			else
			{
				probs[j] = exp(probs[j]);
			}
		}
	}

	return probs;
}


vector<double> handle_underflow_old(vector<double> probs)
{
	int num = probs.size();

	double mult_const = 0.0;

	int less = 0;

	for(int j = 0; j < num; ++j)
	{


		probs[j] = probs[j]-MIN_LOG;


		if(probs[j] < MIN_LOG)
		{
			probs[j] = 0;
		}

		else probs[j] = exp(probs[j]);
		
	}

	return probs;
}

#endif