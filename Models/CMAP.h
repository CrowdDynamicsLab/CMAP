#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <bits/stdc++.h>
#include <random>
#include <tr1/cmath>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>

using namespace std;

typedef struct interaction{
	vector<int> text;
	vector<int> y_ind;
	int behav;
	int user_id;
	double ts;
	vector<int> links;
	vector<int> links_rev;
}interaction;


typedef struct sample_table_return{
	int sampled_table;
	int new_table;
	int sampled_group;
	int sampled_b_topic;
	vector<int> sampled_word_topics;
}sample_table_return;


double MIN_LOG = log(DBL_MIN);
int RESOLUTION = 100;
int num_top_words = 20;
int num_top_users = 20;


int num_init_iter = 10;
int numIter = 500;