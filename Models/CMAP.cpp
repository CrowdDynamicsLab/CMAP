#include "utils.h"
#include "CMAP_UNIFIED.h"
#include "CMAP_FACTORED.h"
#include "CMAP_UNIFIED_HR.h"
#include "CMAP_FACTORED_HR.h"
#include "CMAP_FACTORED_TH.h"
#include "CMAP_FACTORED_HR_TH.h"

#include <signal.h>

using namespace std;

int model = 0;
int hr = 0;
int t = 0;
int G = 20;
int K_w = 20;
int K_b = 5;
int K = 20;
double scale = 1.5;
double discount = 0.5;
int NUM_THREADS = 16;

string dataset = "";

void print_help()
{
	cout <<
		"--dataset <name>:         Name of the dataset to use (Mandatory)\n"
		"--model <type>:           Model-Type. 0 for unified, 1 for factored (default 0)\n"
		"--hr:                     Use the hierarachical version of the model\n"
		"--thread <num_thread>:    Use the threaded version of the model. Specify number of threads to use\n"
		"--G <num_groups>:         Specify the value of number of groups (default: 20)\n"
		"--K <num_topics>:         Specify number of topics (Use only for unified model) (default: 20)\n"
		"--K_w <num_text_topics>:  Specify number of text topics (Use only for factored model) (default: 20)\n"
		"--K_b <num_behav_topics>: Specify number of behavior topics (Use only for factored model) (default: 5)\n"
		"--scale <s>:              Specify the value of scale parameter (default: 1.5)\n"
		"--discount <d>:           Specify the value of dicount parameter (default: 0.5)\n"
		"--iter <num_iter>:        Specify the number of iterations to run (default: 500)\n"
		"--help:                   Print help\n"
		;

	exit(-1);
}

void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    cout<<"Invalid Argument:"<<endl;

    cout<<
    	"--dataset <name>:         Name of the dataset to use (Mandatory)\n"
    	"--model <type>:           Model-Type. 0 for unified, 1 for factored (default 0)\n"
		"--hr:                     Use the hierarachical version of the model\n"
		"--thread <num_thread>:    Use the threaded version of the model. Specify number of threads to use\n"
		"--G <num_groups>:         Specify the value of number of groups (default: 20)\n"
		"--K <num_topics>:         Specify number of topics (Use only for unified model) (default: 20)\n"
		"--K_w <num_text_topics>:  Specify number of text topics (Use only for factored model) (default: 20)\n"
		"--K_b <num_behav_topics>: Specify number of behavior topics (Use only for factored model) (default: 5)\n"
		"--scale <s>:              Specify the value of scale parameter (default: 1.5)\n"
		"--discount <d>:           Specify the value of dicount parameter (default: 0.5)\n"
		"--iter <num_iter>:        Specify the number of iterations to run (default: 500)\n"
		"--help:                   Print help\n"
		;
    exit(-1);
}


void segfault_sigaction_1(int signal, siginfo_t *si, void *arg)
{
	cout<<"Segmentation fault (core dumped)"<<endl;
	exit(0);
}

template <typename C>

void run_CMAP(C cmap)
{
	string destpath = "";

	if(model == 0) destpath = "../Output/"+cmap.model_type+"_"+to_string(K)+"_"+to_string(G)+"_"+dataset+"/";

	else  destpath = "../Output/"+cmap.model_type+"_"+to_string(K_b)+"_"+to_string(G)+"_"+dataset+"/";
	
	make_dir(destpath);

	string basepath = "../Data/";
	string corpus_path= basepath+dataset+"_pre_processed.txt";
	string link_path = basepath+dataset+"_links.txt";

	cmap.initialize(corpus_path.c_str(), link_path.c_str());

	cmap.output_data_stats();

	cmap.run_topic_model(destpath,dataset);

	cmap.output_data_stats();

	cout<<"Writing Result Started"<<endl;

	cmap.output_result(destpath);

	cout<<"Writing Results Over"<<endl;
}

int main(int argc, char **argv)
{
	struct sigaction sa;
	memset(&sa, 0, sizeof(struct sigaction));
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = segfault_sigaction;
	sa.sa_flags   = SA_SIGINFO;

	sigaction(SIGSEGV, &sa, NULL);

	int c;
	int digit_optind = 0;

	while(1) {
		int this_option_optind = optind ? optind : 1;
		int option_index = 0;
		static struct option long_options[] = {
			{"dataset", required_argument, 0,  0 },
			{"model", required_argument, 0,  1 },
			{"hr",  0, 0,  2 },
			{"thread",  required_argument, 0,  3 },
			{"G", required_argument, 0,  4 },
			{"K",  required_argument, 0, 5},
			{"K_w",    required_argument, 0,  6 },
			{"K_b",    required_argument, 0,  7 },
			{"scale",    required_argument, 0,  8},
			{"discount",    required_argument, 0,  9},
			{"iter",    required_argument, 0,  10},
			{"help", 0, 0, 11}
		};

		c = getopt_long(argc, argv, "", long_options, &option_index);
		if (c == -1) break;

		switch (c) {
			case 0:{
				string data(optarg);
				dataset = data;
				break;
			}

			case 1:
				model = stoi(optarg);
				break;

			case 2:
				hr = 1;
				break;

			case 3:
				t = 1;
				NUM_THREADS = stoi(optarg);
				break;

			case 4:
				G = stoi(optarg);
				break;

			case 5:
				K = stoi(optarg);
				break;

			case 6:
				K_w = stoi(optarg);
				break;

			case 7:
				K_b = stoi(optarg);
				break;

			case 8:
				scale = stod(optarg);
				break;

			case 9:
				discount = stod(optarg);
				break;

			case 10:
				numIter = stoi(optarg);
				break;

			case 11:
				print_help();
				break;

			default:
				cout<<"Invalid Argument, Usage: "<<argv[0]<<endl;
				print_help();
				break;
		}
	}
	

	if(dataset == "")
	{
		cout<<"Dataset not provided. Usage: "<<argv[0]<<endl;
		print_help();
		exit(-1);
	}

	sa.sa_sigaction = segfault_sigaction_1;
	sa.sa_flags   = SA_SIGINFO;

	sigaction(SIGSEGV, &sa, NULL);

	if(model==0)
	{
		cout<<"Model-Type: Unified, G: "<<G<<", K: "<<K<<", scale: "<<scale<<", discount: "<<discount<<endl;
	}
	else
	{
		cout<<"Model-Type: Factored, G: "<<G<<", K_w: "<<K_w<<", K_b: "<<K_b<<", scale: "<<scale<<", discount: "<<discount<<endl;
	}


	if(hr == 1)
	{
		cout<<"Using Hierarachical Version ";
	}

	else
	{
		cout<<"Using Non-Hierarchical Version ";
	}

	if(t == 1)
	{
		cout<<"With threading, Number of Threads: "<<NUM_THREADS<<endl;
	}

	else
	{
		cout<<"Without threading"<<endl;
	}

	if(model == 0)
	{
		if(hr == 0)
		{
			if(t == 0)
			{
				CMAP_UNIFIED cmap(G, K, discount, scale);
				run_CMAP<CMAP_UNIFIED>(cmap);
			}

			else
			{
				CMAP_UNIFIED cmap(G, K, discount, scale);
				run_CMAP<CMAP_UNIFIED>(cmap);
			}
		}

		else
		{
			if(t == 0)
			{
				CMAP_UNIFIED_HR cmap(G, K, discount, scale);
				run_CMAP<CMAP_UNIFIED_HR>(cmap);
			}

			else
			{
				CMAP_UNIFIED_HR cmap(G, K, discount, scale);
				run_CMAP<CMAP_UNIFIED_HR>(cmap);
			}
		}
	}

	else
	{
		if(hr == 0)
		{
			if(t == 0)
			{
				CMAP_FACTORED cmap(G, K_b, K_w, discount, scale);
				run_CMAP<CMAP_FACTORED>(cmap);
			}

			else
			{
				CMAP_FACTORED_TH cmap(G, K_b, K_w, discount, scale, NUM_THREADS);
				run_CMAP<CMAP_FACTORED_TH>(cmap);
			}
		}

		else
		{
			if(t == 0)
			{
				CMAP_FACTORED_HR cmap(G, K_b, K_w, discount, scale);
				run_CMAP<CMAP_FACTORED_HR>(cmap);
			}

			else
			{
				CMAP_FACTORED_HR_TH cmap(G, K_b, K_w, discount, scale, NUM_THREADS);
				run_CMAP<CMAP_FACTORED_HR_TH>(cmap);
			}
		}
	}

	return 0;
}