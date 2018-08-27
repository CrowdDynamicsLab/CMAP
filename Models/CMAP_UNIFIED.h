#include "utils.h"

using namespace std;


class CMAP_UNIFIED{

public:
	string model_type = "CMAP_UNIFIED";

	// Model Parameters
	double alpha_u = 0.01;
	double alpha_z = 0.01;		
	double alpha_w = 0.01;
	double alpha_b = 0.01;			// Changed Later to 50/B
	double alpha_gt = 0.01;			// Changed Later to 50/G
	double alpha_gk = 0.01;		// Changed Later to 50/K
	double scale = 1.5;
	double discount = 0.5;
	double lambda_0 = 0.5;
	double lambda_1 = 0.5;

	int K;
	int G;

	vector<vector<double>> alpha_g;
	vector<vector<double>> beta_g;

	// Behavior and Word Mappings to Idx
	map<string,int> wordMap;
	map<int,string> wordMapRev;
	map<string,int> behavMap;
	map<int,string> behavMapRev;


	// Vocabulary and Behavior Set
	set<string> vocab;
	set<string> behav_set;
	set<int> user_set;

	// Dataset
	vector<interaction> corpus;	// List of Interactions in the courpus
	vector<vector<int>> user_intr_list;
	vector<vector<int>> USER_BEHAV_CNT;


	int V;						// Size of Vocabulary
	int B;						// Number of Behavior
	int U;						// Number of Users
	int C;						// Number of Interactions (Size of the corpus)
	int T;						// Number of Tables

	// Count Matrices
	vector<vector<int>> ngu;	// number of times group g is assigned to user u (G x U)
	vector<vector<int>> nkb;	// number of times topic k is assigned to behavior b (K x B)
	vector<vector<int>> nkw;	// number of times topic k is assigned to word w (K x V)
	// vector<vector<int>> ntgk_w;	// number of tables to which group g and topic k are assigned (G x K)
	vector<vector<int>> ngk;	// number of interactions to which group g and topic k are assigned (G x K)	
	vector<vector<int>> n_links;// number of links from group g to group g' (G x G)

	vector<int> ngksum;

	// Probability Matrices
	vector<vector<double>> PHI_G_U_num;
	vector<vector<double>> PHI_K_B_num;
	vector<vector<double>> PHI_K_W_num;
	// vector<vector<double>> LINK_PROB_num;

	vector<double> PHI_G_U_denom;
	vector<double> PHI_K_B_denom;
	vector<double> PHI_K_W_denom;


	vector<double> g_const;
	vector<double> g_const_num;
	vector<vector<int>> g_tables;

	// Count Arrays
	vector<int> ng;				// number of users(/interactions) assigned to group g (G x 1)
	vector<int> nkbsum;			// number of behaviors(/interactions) assigned to topic k (K x 1)
	vector<int> nkwsum;			// number of words(/tokens) assigned to topic k (K x 1)
	vector<int> n_table;		// number of interactions on table i (T x 1)
	vector<int> ntg;			// number of tables to which group g is assigned  (G x 1)

	// Probability Matrices
	vector<vector<vector<double>>> pgkt;	// Beta Distribution probability of the group g at time t (G x RESOLUTION)

	vector<double> PY_PROB;		// Pitman-Yor Probability Term for all possible counts
	vector<double> GROUP_PRIOR;

	vector<vector<double>> GROUP_G_K_PRIOR_NUM;
	vector<double> GROUP_G_K_PRIOR_DENOM;

	// Assignments
	vector<int> table_group;	// Group assigned to table i (T x 1)
	vector<int> user_table;		// Table on which interaction i is
	vector<int> user_group;		// Group assigned to interactions (Used only in the 1st iteraction)

	vector<int> intr_topic;	// Topic assigned to interactions (Used only in the 1st iteraction)
	vector<int> intr_group;

	vector<int> empty_tables;		// Empty table list



	// Filenames
	string vocab_map_filename = "vocab-mapping.txt";
	string behavior_map_filename = "behavior-mapping.txt";
	string group_prior_filename = "group-priors.txt";
	string group_topic_filename = "group-topic-distribution.txt";
	string group_user_filename = "group-user-distribution.txt";
	string group_time_alpha_filename = "group-time-alpha.txt";
	string group_time_beta_filename = "group-time-beta.txt";
	string topic_time_alpha_filename = "topic-time-alpha.txt";
	string topic_time_beta_filename = "topic-time-beta.txt";
	string topic_word_filename = "topic-word-distribution.txt";
	string topic_behav_filename = "topic-behavior-distribution.txt";
	string table_seating_filename = "table-assignment-status.txt";
	string top_topic_words_filename = "top-topic-words.txt";
	string top_topic_behav_filename = "top-topic-behav.txt";
	string top_group_topic_filename = "top-group-topic.txt";
	string top_group_user_filename = "top-group-users.txt";
	string link_prob_filename = "link-prob.txt";
	string g_k_tables_filename = "g_k_tables.txt";

	double step_1_sum = 0.0;
	double step_2_sum = 0.0;
	double step_3_sum = 0.0;
	double step_4_sum = 0.0;
	double step_5_sum = 0.0;
	double step_6_sum = 0.0;
	double step_7_sum = 0.0;
	double step_8_sum = 0.0;

	vector<double> max_values; 


public:

	CMAP_UNIFIED(int _G = 20, int _K = 20, int _discount = 0.5, int _scale = 1.5)
	{
		this->G = _G;
		this->K = _K;
		this->discount = _discount;
		this->scale = _scale;
	}


	double compute_p_i_k(int itr, int i, int k)
	{
		double p = 0.0;

		interaction curr_intr = corpus[i];

		int u = curr_intr.user_id;
		int b = curr_intr.behav;

		p = p + log((nkb[k][b] + alpha_b) / (nkbsum[k] + (B * alpha_b)));

		int W = curr_intr.text.size();

		for(int j = 0; j <  W; ++j)
		{
			int w = curr_intr.text[j];
			p = p + log((nkw[k][w] + alpha_w)/(nkwsum[k] + (V * alpha_w)));
		}

		return p;
	}


	void seat_to(int itr, int u, int table, vector<vector<double>> p_k)
	{

		clock_t begin_time = clock();
		n_table[table] += 1;

		int g = table_group[table];

		user_group[u] = g;
		user_table[u] = table;

		ngu[g][u] += 1;
		ng[g] += 1;

		PHI_G_U_num[g][u] = log((ngu[g][u] + alpha_u));
		PHI_G_U_denom[g] = log((ng[g] + (U * alpha_u)));

		int num_interactions = user_intr_list[u].size();

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			// Sample Word Topics 
			vector<double> prob_distr_k;

			prob_distr_k.resize(K, 0.0);

			for(int k = 0; k < K; k++)
			{
				prob_distr_k[k] = (GROUP_G_K_PRIOR_NUM[g][k] - GROUP_G_K_PRIOR_DENOM[g]) + p_k[i][k];
			}

			prob_distr_k = handle_underflow(prob_distr_k);

			int sampled_topic = sample_from_prob(prob_distr_k);

			intr_topic[user_intr_list[u][i]] = sampled_topic;

		}

		clock_t step_7_time = clock();

		step_7_sum += double(step_7_time - begin_time) / CLOCKS_PER_SEC;

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			int k = intr_topic[user_intr_list[u][i]];

			int b = curr_intr.behav;

			nkb[k][b] += 1;
			nkbsum[k] += 1;

			ngk[g][k] += 1;
			ngksum[g] += 1;


			GROUP_G_K_PRIOR_NUM[g][k] = log((ngk[g][k] + alpha_gk));
			GROUP_G_K_PRIOR_DENOM[g] = log((ngksum[g] + K * alpha_gk));

			int W = curr_intr.text.size();

			for(int j = 0; j <  W; ++j)
			{
				int w = curr_intr.text[j];

				nkw[k][w] += 1;
				nkwsum[k] += 1;	

				PHI_K_W_num[k][w] = log((nkw[k][w] + alpha_w));
			}

			intr_topic[user_intr_list[u][i]] = k;

			PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));
			PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));

			PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));
		}

		clock_t step_8_time = clock();

		step_8_sum += double(step_8_time - step_7_time) / CLOCKS_PER_SEC;

		// Increase dominance cnts

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];
			int num_links = curr_intr.links.size();

			for(int l = 0; l < num_links; l++)
			{
				int j = intr_group[curr_intr.links[l]];

				n_links[g][j] += 1;

				if(n_links[g][j] < 0)
				{
					cout<<"#### ERROR ####### After: "<<i<<", "<<curr_intr.links[l]<<endl;
				}
			}

			int num_links_rev = curr_intr.links_rev.size();

			for(int l = 0; l < num_links_rev; l++)
			{
				int j = intr_group[curr_intr.links_rev[l]];

				n_links[j][g] += 1;
			}

			intr_group[user_intr_list[u][i]] = g;
		}

	}

	void increase_cnts(int itr, int u, int g)
	{
		user_group[u] = g;

		ngu[g][u] += 1;
		ng[g] += 1;

		vector<int> behav_cnts = USER_BEHAV_CNT[u];

		
		PHI_G_U_num[g][u] = log((ngu[g][u] + alpha_u));
		PHI_G_U_denom[g] = log((ng[g] + (U * alpha_u)));

		int num_interactions = user_intr_list[u].size();

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			// Sample Word Topics 
			vector<double> prob_distr_k;

			prob_distr_k.resize(K, 0.0);

			for(int k = 0; k < K; k++)
			{
				double ts = curr_intr.ts;

				int t = int(ts*RESOLUTION);
				
				prob_distr_k[k] = (log((ngk[g][k] + alpha_gk) / (ng[g] + K * (alpha_gk)))) + compute_p_i_k(itr, user_intr_list[u][i], k) + pgkt[g][k][t];
			}

			prob_distr_k = handle_underflow(prob_distr_k);

			int sampled_topic = sample_from_prob(prob_distr_k);

			intr_topic[user_intr_list[u][i]] = sampled_topic;

		}

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			if(itr==0) intr_group[user_intr_list[u][i]] = g;

			int k = intr_topic[user_intr_list[u][i]];

			int b = curr_intr.behav;

			nkb[k][b] += 1;
			nkbsum[k] += 1;

			ngk[g][k] += 1;
			ngksum[g] += 1;

			int W = curr_intr.text.size();

			for(int j = 0; j <  W; ++j)
			{
				int w = curr_intr.text[j];

				nkw[k][w] += 1;
				nkwsum[k] += 1;

				PHI_K_W_num[k][w] = log((nkw[k][w] + alpha_w));
			}

			intr_topic[user_intr_list[u][i]] = k;

			PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));

			PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
			PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));
		}


		if(itr != 0)
		{
			for(int i = 0; i < num_interactions; i++)
			{
				interaction curr_intr = corpus[user_intr_list[u][i]];
				int num_links = curr_intr.links.size();

				for(int l = 0; l < num_links; l++)
				{
					int j = intr_group[curr_intr.links[l]];

					n_links[g][j] += 1;

					if(n_links[g][j] < 0)
					{
						cout<<"#### ERROR-I ####### After: "<<i<<", "<<curr_intr.links[l]<<endl;

						int total = 0;
						for(int g = 0; g < G; ++g)
						{
							for(int j = 0; j< G; ++j)
							{
								total+= n_links[g][j];
								cout<<n_links[g][j]<<" ";
							}

							cout<<endl;
						}

						cout<<"Total: "<<total<<endl;
					}
				}

				int num_links_rev = curr_intr.links_rev.size();

				for(int l = 0; l < num_links_rev; l++)
				{
					int j = intr_group[curr_intr.links_rev[l]];

					n_links[j][g] += 1;
				}

				intr_group[user_intr_list[u][i]] = g;
			}
		}
	}


	void decrease_cnts(int itr, int u)
	{
		int g = user_group[u];

		ngu[g][u] -= 1;
		ng[g] -= 1;


		PHI_G_U_num[g][u] = log((ngu[g][u] + alpha_u));
		PHI_G_U_denom[g] = log((ng[g] + (U * alpha_u)));


		int num_interactions = user_intr_list[u].size();

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			int k = intr_topic[user_intr_list[u][i]];

			int b = curr_intr.behav;

			ngk[g][k] -= 1;
			ngksum[g] -= 1;

			nkb[k][b] -= 1;
			nkbsum[k] -= 1;

			// GROUP_G_W_PRIOR_NUM[g][k_w] = log((ngk_w[g][k_w] + alpha_gtk_w));
			// GROUP_G_W_PRIOR_DENOM[g] = log((ngk_wsum[g] + K_w * alpha_gtk_w));

			int W = curr_intr.text.size();

			for(int j = 0; j <  W; ++j)
			{
				int w = curr_intr.text[j];

				nkw[k][w] -= 1;
				nkwsum[k] -= 1;

				PHI_K_W_num[k][w] = log((nkw[k][w] + alpha_w));
			}

			intr_topic[user_intr_list[u][i]] = -1;

			PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));

			PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
			PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));
		}


		// // Decrease dominance cnts


		if(itr != 0)
		{
			for(int i = 0; i < num_interactions; i++)
			{
				interaction curr_intr = corpus[user_intr_list[u][i]];
				int num_links = curr_intr.links.size();

				for(int l = 0; l < num_links; l++)
				{
					int j = intr_group[curr_intr.links[l]];

					n_links[g][j] -= 1;

					if(n_links[g][j] < 0)
					{
						cout<<"#### ERROR-D ####### After: "<<i<<", "<<curr_intr.links[l]<<endl;

						int total = 0;
						for(int g = 0; g < G; ++g)
						{
							for(int j = 0; j< G; ++j)
							{
								total+= n_links[g][j];
								cout<<n_links[g][j]<<" ";
							}

							cout<<endl;
						}

						cout<<"Total: "<<total<<endl;

					}
				}

				int num_links_rev = curr_intr.links_rev.size();

				for(int l = 0; l < num_links_rev; l++)
				{
					int j = intr_group[curr_intr.links_rev[l]];

					n_links[j][g] -= 1;
				}

				// intr_group[user_intr_list[u][i]] = -1;
			}
		}


		user_group[u] = -1;
	}


	void unseat_initial(int itr, int u)
	{

		int g = user_group[u];

		ngu[g][u] -= 1;
		ng[g] -= 1;

		PHI_G_U_num[g][u] = log((ngu[g][u] + alpha_u));
		PHI_G_U_denom[g] = log((ng[g] + (U * alpha_u)));


		int num_interactions = user_intr_list[u].size();

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			int k = intr_topic[user_intr_list[u][i]];

			int b = curr_intr.behav;

			ngk[g][k] -= 1;
			ngksum[g] -= 1;

			nkb[k][b] -= 1;
			nkbsum[k] -= 1;

			GROUP_G_K_PRIOR_NUM[g][k] = log((ngk[g][k] + alpha_gk));
			GROUP_G_K_PRIOR_DENOM[g] = log((ngksum[g] + K * alpha_gk));

			int W = curr_intr.text.size();

			for(int j = 0; j <  W; ++j)
			{
				int w = curr_intr.text[j];

				nkw[k][w] -= 1;
				nkwsum[k] -= 1;

				PHI_K_W_num[k][w] = log((nkw[k][w] + alpha_w));
			}

			intr_topic[user_intr_list[u][i]] = -1;

			PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));

			PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
			PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));
		}


		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			int num_links = curr_intr.links.size();

			for(int l = 0; l < num_links; l++)
			{
				int j = intr_group[curr_intr.links[l]];

				n_links[g][j] -= 1;

				if(n_links[g][j] < 0)
				{
					cout<<"#### ERROR ####### After: "<<i<<", "<<curr_intr.links[l]<<endl;
				}
			}

			int num_links_rev = curr_intr.links_rev.size();

			for(int l = 0; l < num_links_rev; l++)
			{
				int j = intr_group[curr_intr.links_rev[l]];

				n_links[j][g] -= 1;
			}

			// intr_group[user_intr_list[u][i]] = -1;
		}

		user_group[u] = -1;
	}

	void unseat_from(int itr, int u, int table)
	{
		n_table[table] -= 1;

		int g = table_group[table];

		ngu[g][u] -= 1;
		ng[g] -= 1;


		PHI_G_U_num[g][u] = log((ngu[g][u] + alpha_u));
		PHI_G_U_denom[g] = log((ng[g] + (U * alpha_u)));


		int num_interactions = user_intr_list[u].size();

		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			int k = intr_topic[user_intr_list[u][i]];

			int b = curr_intr.behav;

			ngk[g][k] -= 1;
			ngksum[g] -= 1;

			nkb[k][b] -= 1;
			nkbsum[k] -= 1;

			GROUP_G_K_PRIOR_NUM[g][k] = log((ngk[g][k] + alpha_gk));
			GROUP_G_K_PRIOR_DENOM[g] = log((ngksum[g] + K * alpha_gk));

			int W = curr_intr.text.size();

			for(int j = 0; j <  W; ++j)
			{
				int w = curr_intr.text[j];
			
				nkw[k][w] -= 1;
				nkwsum[k] -= 1;

				PHI_K_W_num[k][w] = log((nkw[k][w] + alpha_w));
			}

			intr_topic[user_intr_list[u][i]] = -1;

			PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));

			PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
			PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));
		}

		// Decrease dominance cnts


		for(int i = 0; i < num_interactions; i++)
		{
			interaction curr_intr = corpus[user_intr_list[u][i]];

			int num_links = curr_intr.links.size();

			for(int l = 0; l < num_links; l++)
			{
				int j = intr_group[curr_intr.links[l]];

				n_links[g][j] -= 1;

				if(n_links[g][j] < 0)
				{
					cout<<"#### ERROR ####### After: "<<i<<", "<<curr_intr.links[l]<<endl;
				}
			}

			int num_links_rev = curr_intr.links_rev.size();

			for(int l = 0; l < num_links_rev; l++)
			{
				int j = intr_group[curr_intr.links_rev[l]];

				n_links[j][g] -= 1;
			}

			// intr_group[user_intr_list[u][i]] = -1;
		}


		if(n_table[table] == 0)
		{
			// Update empty tables
			empty_tables.push_back(table);
			// cout<<"Empty Table Added: "<<table<<endl;
			// Decrease the number of groups and number of (group, topic) counts

			ntg[table_group[table]] -= 1;

			int new_group = table_group[table];

			// Update Group Prior and Group_Topic Prior

			if(itr != 0)
			{

				GROUP_PRIOR[new_group] = log((ntg[new_group] + alpha_gt) / ((T - empty_tables.size()) + G * alpha_gt));
			}

			// Make the table invalid
			table_group[table] = -1;

			// Update g_const
			if(itr != 0) g_const[g] -= ((1.0 - discount)/ (U - 1 + scale));

			else g_const_num[g] -= (1.0 - discount);

			// Remove the table from g_k_tables
			g_tables[g].erase(remove(g_tables[g].begin(), g_tables[g].end(), table), g_tables[g].end());
		}

		else
		{
			// Update g_k_const
			if(itr!=0) g_const_num[g] -= (1.0 / (U - 1 + scale));

			else g_const_num[g] -= (1.0);
		}

		user_group[u] = -1;
	}


	double dominance_prob(int i, int g)
	{
		double prob = 0.0;

		interaction curr_intr = corpus[i];

		int num_links = curr_intr.links.size();

		for(int l = 0; l < num_links; l++)
		{
			int j = intr_group[curr_intr.links[l]];

			// cout<<j<<" "<<g<<endl;

			prob += log((n_links[g][j] + lambda_0) / (n_links[g][j] + n_links[j][g] + lambda_0 + lambda_1));
		}

		int num_rev_links = curr_intr.links_rev.size();

		for(int l = 0; l < num_rev_links; l++)
		{
			int j = intr_group[curr_intr.links_rev[l]];

			// cout<<j<<" "<<g<<endl;

			prob += log((n_links[j][g] + lambda_0) / (n_links[j][g] + n_links[g][j] + lambda_0 + lambda_1));
		}

		return prob;
	}



	double compute_p_u_g(int itr, int u, int g)
	{
		double p = 0.0;

		p = p + PHI_G_U_num[g][u] - PHI_G_U_denom[g];

		return p;
	}


	void gibbs_sampling_iteration(int itr, clock_t last_time, string destpath)
	{
		int N = 0;			// Number of interactions already seated on the table


		double itr_secs = 0.0;

		double unseat_avg = 0.0;
		double sample_avg = 0.0;
		double seating_avg = 0.0;

		double total_secs = 0.0;

		step_1_sum = 0.0;
		step_2_sum = 0.0;
		step_3_sum = 0.0;
		step_4_sum = 0.0;
		step_5_sum = 0.0;
		step_6_sum = 0.0;
		step_7_sum = 0.0;
		step_8_sum = 0.0;

		ofstream user_posterior_file;
		ofstream user_posterior_temp_file;
		if(itr == numIter-1)
		{
			user_posterior_file.open((destpath+"user_posteriors.txt").c_str(), ofstream::out);
			user_posterior_temp_file.open((destpath+"user_posteriors_temp.txt").c_str(), ofstream::out);
		}

		for(int u = 0; u < U; ++u)
		{

			if(itr == 0)
			{
				N = u+1;
			}

			else
			{
				N = U;
			}

			clock_t last = clock();

			if(itr != 0)
			{
				unseat_from(itr, u, user_table[u]);
			}

			else
			{
				unseat_initial(itr, u);
			}

			clock_t unseat_time = clock();

			double unseat_secs = double(unseat_time - last) / CLOCKS_PER_SEC;

			clock_t begin_time = clock();
			// Compute TS-Term & Word-Topic Term
			vector<vector<double>> p_k;
		
			vector<double> topic_term;

			topic_term.resize(G, 0.0);

			int num_interactions = user_intr_list[u].size();

			p_k.resize(num_interactions);

			clock_t step_1_time = clock();

			step_1_sum += double(step_1_time - begin_time) / CLOCKS_PER_SEC;


			for(int i = 0; i < num_interactions; i++)
			{
				interaction curr_intr = corpus[user_intr_list[u][i]];

				vector<double> curr_p_k;
				curr_p_k.resize(K, 0.0);
				p_k[i].resize(K, 0.0);

				for(int k = 0; k < K; ++k)
				{
					p_k[i][k] = compute_p_i_k(itr, user_intr_list[u][i], k);
					curr_p_k[k] = p_k[i][k];
				}

				curr_p_k = handle_underflow(curr_p_k);

				double max_prob = *max_element(curr_p_k.begin(), curr_p_k.begin()+K);

				vector<double> p_g_k;
				p_g_k.resize(G, 0.0);

				for(int g = 0; g < G; g++)
				{
					for(int k = 0; k < K; ++k)
					{	
						double ts = curr_intr.ts;

						int t = int(ts*RESOLUTION);
						p_g_k[g] += (exp(GROUP_G_K_PRIOR_NUM[g][k] - GROUP_G_K_PRIOR_DENOM[g])) * curr_p_k[k] * exp(pgkt[g][k][t]); 
					}

					if(p_g_k[g] == 0)
					{
						p_g_k[g] = max_prob;
					}

					else
					{
						p_g_k[g] = log(p_g_k[g])+max_prob;
					}

					topic_term[g] += p_g_k[g];
				}
			}

			clock_t step_2_time = clock();

			step_2_sum += double(step_2_time - step_1_time) / CLOCKS_PER_SEC;


			clock_t step_3_time = clock();

			step_3_sum += double(step_3_time - step_2_time) / CLOCKS_PER_SEC;

			// Compute Links Term

			vector<double> links_prob;
			links_prob.resize(G, 0.0);

			for(int i = 0; i < num_interactions; i++)
			{
				interaction curr_intr = corpus[user_intr_list[u][i]];

				for(int g = 0; g < G; ++g)
				{
					links_prob[g] += dominance_prob(user_intr_list[u][i], g);
				}
			}


			clock_t step_4_time = clock();

			step_4_sum += double(step_4_time - step_3_time) / CLOCKS_PER_SEC;

			// Sample Table

			int sampled_table = -1;
			vector<double> p;			// Probabilities of placing interaction i on different tables
			int new_table = -1;

			// Sample G & K_b

			vector<double> prob_distr_g;

			prob_distr_g.resize(G, -1.0);


			for(int g = 0; g < G; ++g)
			{


				prob_distr_g[g] = 0.0;

				prob_distr_g[g] += topic_term[g];

				prob_distr_g[g] += links_prob[g];

				if(itr != 0)
				{
					prob_distr_g[g] += log(g_const[g] + (((scale + (T - empty_tables.size())*discount) / (N - 1 + scale)) * exp(GROUP_PRIOR[g]) ));
				}

				else
				{
					prob_distr_g[g] += log((g_const_num[g] / (N - 1 + scale)) + (((scale + (T - empty_tables.size())*discount) / (N - 1 + scale)) * exp(GROUP_PRIOR[g]) ));
				}
			}

			vector<double> prob_distr_g_temp = prob_distr_g;

			prob_distr_g = handle_underflow(prob_distr_g);
			int sampled_group = sample_from_prob(prob_distr_g);

			clock_t step_5_time = clock();

			step_5_sum += double(step_5_time - step_4_time) / CLOCKS_PER_SEC;


			if(empty_tables.empty())
			{
				new_table = T;
			}

			else
			{
				new_table = empty_tables[empty_tables.size()-1];	// The last empty table as the new table
			}


			vector<int> table_li = g_tables[sampled_group];			// Tables already assigned this (g,k) pair
			int num_tables = table_li.size();

			table_li.push_back(new_table);

			vector<double> table_prob_distr;

			table_prob_distr.resize(num_tables+1, 0.0);

			for(int j = 0; j < num_tables; j++)
			{
				int t = table_li[j];

				if(itr != 0) table_prob_distr[j] = PY_PROB[n_table[t]];
				
				else table_prob_distr[j] = log((n_table[t] - discount)/ (N - 1 + scale));
			}

			table_prob_distr[num_tables] = log((scale + (T - empty_tables.size()) * discount) / (N - 1 + scale)) + GROUP_PRIOR[sampled_group];

			if(num_tables == 0)
			{
				sampled_table = new_table;
			}

			else
			{

				table_prob_distr = handle_underflow(table_prob_distr);

				sampled_table = sample_from_prob(table_prob_distr);

				sampled_table = table_li[sampled_table];
			}

			
			clock_t step_6_time = clock();

			step_6_sum += double(step_6_time - step_5_time) / CLOCKS_PER_SEC;


			if(sampled_table == new_table)
			{
				// Assign Group and topic to the new table
				if(new_table == T)
				{
					table_group.push_back(sampled_group);
					T+=1;

					n_table.resize(T);

					n_table[T-1] = 0;

					// cout<<"New Table Added: "<<new_table<<", g: "<<sampled_group<<", k: "<<sampled_b_topic<<endl;
				}
				else
				{
					table_group[new_table] = sampled_group;

					// Delete the old empty table from empty tables list
					empty_tables.pop_back();

					// cout<<"Empty Table Removed: "<<new_table<<", g: "<<sampled_group<<", k: "<<sampled_b_topic<<endl;
				}

				// Increase the group and (group, topic) cnt
				ntg[sampled_group] += 1;

				// Update Group Prior and Group_Topic Prior

				if(itr != 0)
				{

					GROUP_PRIOR[sampled_group] = log((ntg[sampled_group] + alpha_gt) / ((T - empty_tables.size()) + G * alpha_gt));

					g_const[sampled_group] += ( (1.0 - discount )/ (U - 1 + scale));
				}

				else
				{
					g_const_num[sampled_group] += (1.0 - discount);
				}

				g_tables[sampled_group].push_back(new_table);

			}

			else
			{
				if(itr != 0) g_const[sampled_group] += ( (1.0)/ (U - 1 + scale));

				else g_const_num[sampled_group] += (1.0);
			}

			clock_t sample_time = clock();
			double sample_secs = double(sample_time - unseat_time) / CLOCKS_PER_SEC;

			seat_to(itr, u, sampled_table, p_k);			// Update Table assignment and associated count matrices and arrays

			clock_t seating_time = clock();
			double seating_secs = double(seating_time - sample_time) / CLOCKS_PER_SEC;

			unseat_avg += unseat_secs;
			seating_avg += seating_secs;
			sample_avg += sample_secs;
			// Seating Completed

			if(itr == numIter-1)
			{
				for(int g = 0; g < G; g++){
					user_posterior_file<<prob_distr_g[g]<<" ";
					user_posterior_temp_file<<prob_distr_g_temp[g]<<" ";
				}
				user_posterior_file<<endl;
				user_posterior_temp_file<<endl;
			}
		}

		if(itr == numIter-1)
		{
			user_posterior_file.close();
			user_posterior_temp_file.close();
		}

		double unseat_sum = unseat_avg;

		double sample_sum = sample_avg;

		double seating_sum = seating_avg;

		unseat_avg = (unseat_avg)/U;
		sample_avg = (sample_avg)/U;
		seating_avg = (seating_avg)/U;

		cout<<"CMAP: Iter: "<<itr<<"  Unseat: "<<unseat_avg<<"/"<<unseat_sum<<", Sample: "<<sample_avg<<"/"<<sample_sum<<", Seating: "<<seating_avg<<"/"<<seating_sum<<endl;
		cout<<"CMAP: Iter: "<<itr<<"  Step 1: "<<(step_1_sum/U)<<"/"<<step_1_sum<<", Step 2 : "<<(step_2_sum/U)<<"/"<<step_2_sum<<", Step 3: "<<(step_3_sum/U)<<"/"<<step_3_sum<<", Step 4: "<<(step_4_sum/U)<<"/"<<step_4_sum<<", Step 5: "<<(step_5_sum/U)<<"/"<<step_5_sum<<", Step 6: "<<(step_6_sum/U)<<"/"<<step_6_sum<<", Step 7: "<<(step_7_sum/U)<<"/"<<step_7_sum<<", Step 8: "<<(step_8_sum/U)<<"/"<<step_8_sum<<", T: "<<(T - empty_tables.size())<<endl;
	}

	void compute_beta_distribution(int itr)
	{

		if(itr != 0)
		{
			map<pair<int, int>, vector<double>> group_ts;

			int idx = 0;
			for(int g = 0; g < G; ++g)
			{
				for(int k = 0; k < K; ++k)
				{
					group_ts[pair<int, int>(g,k)] = vector<double>();
				}
			}

			for(int u = 0; u < U; ++u)
			{

				int curr_group = user_group[u];

				int num_interactions = user_intr_list[u].size();
				for(int i = 0; i < num_interactions; i++)
				{
					interaction curr_intr = corpus[user_intr_list[u][i]];

					int curr_topic = intr_topic[user_intr_list[u][i]];

					double curr_ts = curr_intr.ts;

					group_ts[pair<int,int>(curr_group,curr_topic)].push_back(curr_ts);
				}

			}



			for(int g = 0; g < G; ++g)
			{

				for(int k = 0; k < K; ++k)
				{
					if(group_ts[pair<int, int>(g,k)].size() == 0)
					{
						alpha_g[g][k] = 1.0;
						beta_g[g][k] = 1.0;
						continue;
					}

					if(group_ts[pair<int, int>(g,k)].size()<100)
					{
						continue;
					}

					double t_g = mean(group_ts[pair<int, int>(g,k)]);

					double s_g = standard_deviation(group_ts[pair<int, int>(g,k)], t_g);

					if(s_g != 0)
					{
						double curr_alpha = t_g * ( ((t_g * (1 - t_g)) / (s_g*s_g)) - 1);

						double curr_beta = (1- t_g) * ( ((t_g * (1 - t_g)) / (s_g*s_g)) - 1);

						if(curr_alpha < 200.0 && curr_beta < 200)
						{
							alpha_g[g][k] = curr_alpha;
							beta_g[g][k] = curr_beta;				
						}

					}
				}
			}
		}


		// Compute the probabilities
		for(int g = 0; g < G; ++g)
		{

			for(int k = 0; k < K; ++k)
			{
				for(int t = 1; t < RESOLUTION; ++t)
				{
					double ts = (1.0*t)/(1.0 * RESOLUTION);

					pgkt[g][k][t] = log(pow(ts, alpha_g[g][k]-1)) + log(pow(1-ts, beta_g[g][k]-1)) - log(tr1::beta(alpha_g[g][k], beta_g[g][k]));

				}

			}
		}

	}

	void compute_link_counts()
	{
		for(int i = 0; i < C; ++i)
		{
			interaction curr_intr = corpus[i];

			int group_i = intr_group[i];

			int num_links = curr_intr.links.size();

			for(int l = 0; l < num_links; l++)
			{
				int j = curr_intr.links[l];
				int group_j = intr_group[j];

				n_links[group_i][group_j] += 1;
			}
		}


		cout<<"Printing Links Matrix:"<<endl;
		int total = 0;
		for(int g = 0; g < G; ++g)
		{
			for(int j = 0; j< G; ++j)
			{
				total+= n_links[g][j];
				cout<<n_links[g][j]<<" ";
			}

			cout<<endl;
		}

		cout<<"Total: "<<total<<endl;
	}


	void init_gibbs_sampling_iteration(int itr)
	{
		double decrease_avg = 0.0;
		double sample_avg = 0.0;
		double increase_avg = 0.0;

		for(int u = 0; u< U; ++u)
		{
			clock_t begin_time = clock();
			
			// Decrease the Counts of arrays and matrices
			decrease_cnts(itr, u);

			clock_t decrease_time = clock();
			double decrease_secs = double(decrease_time - begin_time) / CLOCKS_PER_SEC;

			// Sample Group and Topic

			vector<double> p_g;
			vector<double> p_k_b;
			vector<double> p_k_w;

			p_g.resize(G, -1.0);

			vector<double> links_prob;
			links_prob.resize(G, 0.0);
			int num_interactions = user_intr_list[u].size();

			if(itr != 0)
			{
				for(int i = 0; i < num_interactions; i++)
				{
					interaction curr_intr = corpus[user_intr_list[u][i]];

					for(int g = 0; g < G; ++g)
					{
						links_prob[g] += dominance_prob(user_intr_list[u][i], g);
					}
				}
			}

			// Sample Group
			for(int g = 0; g < G; ++g)
			{
				p_g[g] = 0.0;
				p_g[g] += log((ng[g] + alpha_gt) / (U + G * (alpha_gt)));

				p_g[g] += compute_p_u_g(itr, u, g);
				if(itr != 0) p_g[g] += links_prob[g];
			}


			p_g = handle_underflow(p_g);

			int new_group = sample_from_prob(p_g);

			clock_t sample_time = clock();
			double sample_secs = double(sample_time - decrease_time) / CLOCKS_PER_SEC;

			increase_cnts(itr, u, new_group);


			clock_t increase_time = clock();
			double increase_secs = double(increase_time - sample_time) / CLOCKS_PER_SEC;

			decrease_avg += decrease_secs;
			increase_avg += increase_secs;
			sample_avg += sample_secs;
		}
	}

	void run_topic_model(string destpath,string dataset)
	{
		cout<<"Running Topic Model"<<endl;

		double itr_secs = 0.0;
		double total_secs = 0.0;
		double beta_secs = 0.0;
		double total_beta_secs = 0.0;


		for(int itr = 0; itr < num_init_iter; itr++)
		{
			clock_t begin = clock();

			compute_beta_distribution(itr);

			clock_t beta_time = clock();

			beta_secs = double(beta_time - begin) / CLOCKS_PER_SEC;

			total_beta_secs += beta_secs;

			init_gibbs_sampling_iteration(itr);


			if(itr == 0)
			{
				compute_link_counts();
			}


			clock_t end = clock();

			itr_secs = double(end - begin) / CLOCKS_PER_SEC;

			total_secs = total_secs + itr_secs;

			cout<<"\rInitializing Model: "<<itr<<" / "<<num_init_iter;
			fflush(stdout);

		}

		cout<<endl<<"Intitialization Over"<<endl<<endl;

		for(int i = 1; i< U; ++i)
		{
			PY_PROB[i] = log((i - discount)/ (U - 1 +scale));
		}


		for(int g = 0; g < G; ++g)
		{
			GROUP_PRIOR[g] = log((ng[g] + alpha_gt) / (U + G * (alpha_gt)));
		}

		for(int g = 0; g < G; ++g)
		{
			for(int k = 0; k < K; ++k)
			{
				GROUP_G_K_PRIOR_NUM[g][k] = log((ngk[g][k] + alpha_gk));
			}

			GROUP_G_K_PRIOR_DENOM[g] = log((ngksum[g] + K * alpha_gk));
		}


		for(int itr = 0; itr < numIter; itr++)
		{
			clock_t begin = clock();

			compute_beta_distribution(itr+1);

			clock_t beta_time = clock();

			beta_secs = double(beta_time - begin) / CLOCKS_PER_SEC;

			total_beta_secs += beta_secs;

			gibbs_sampling_iteration(itr, beta_time, destpath);

			// cout<<"Gibbs Done"<<endl;
			if(itr == 0)
			{

				for(int g = 0; g < G; ++g)
				{
					GROUP_PRIOR[g] = log((ntg[g] + alpha_gt) / ( (T - empty_tables.size())+ G * (alpha_gt)));
				}

				g_const.resize(G, 0.0);

				for(int t = 0; t < T; ++t)
				{
					if(table_group[t] != -1)
					{
						int g = table_group[t];

						g_const[g] = g_const[g] + ((n_table[t] - discount) / (U - 1 + scale));
					}
				}

			}

			clock_t end = clock();

			itr_secs = double(end - begin) / CLOCKS_PER_SEC;

			total_secs = total_secs + itr_secs;

			cout << "CMAP: Iter: " << itr << '/' << numIter <<"   Last Beta Time: "<<beta_secs<< "   Total Beta Time: "<<total_beta_secs<<"   Last Iter: " << itr_secs << "    Total Time: " << total_secs<<endl<<endl;
		}

		cout<<endl<<"Topic Model Over"<<endl;
	}


	void readVocab(const char* filename)
	{

		int counter = 0;
		int user_id;
		double ts;
		ifstream infile(filename);
		string line;


		while (getline(infile,line)){
			// Text, UserId, Behav, Timestamp
			vector<string> cols = split(line, '\t');
			string text = cols[0];
			stringstream(cols[1]) >> user_id;
			string behav = cols[2];
			stringstream(cols[3]) >> ts;

			vector<string> text_units = split(text, ' ');

			for (int i = 0; i < text_units.size(); i++){
				vocab.insert(text_units[i]);
			}

			behav_set.insert(behav);

			user_set.insert(user_id);
		}

		// Create Word Map
		set<string>::iterator it;
		for (it = vocab.begin(); it != vocab.end(); it++){
			wordMap.insert(pair<string,int>(*it, counter));
			wordMapRev.insert(pair<int,string>(counter, *it));
			counter++;
		}

		// Create Behavior Map
		counter = 0;
		for(it = behav_set.begin(); it != behav_set.end(); it++)
		{
			behavMap.insert(pair<string,int>(*it, counter));
			behavMapRev.insert(pair<int,string>(counter, *it));
			counter++;
		}

	}

	void readCorpus(const char *corpus_path,const char *link_path)
	{
		ifstream infile(corpus_path);
		string line;

		int idx = 0;

		while (getline(infile,line)){
			// Text, UserId, Behav, Timestamp
			interaction curr_intr;
			vector<string> cols = split(line, '\t');
			string text = cols[0];
			stringstream(cols[1]) >> curr_intr.user_id;
			curr_intr.behav = behavMap.find(cols[2])->second;
			stringstream(cols[3]) >> curr_intr.ts;

			curr_intr.text.clear();

			vector<string> text_units = split(text, ' ');

			for (int i = 0; i < text_units.size(); i++){
				curr_intr.text.push_back(wordMap.find(text_units[i])->second);
			}

			curr_intr.y_ind.resize(text_units.size(), -1);

			corpus.push_back(curr_intr);
			user_intr_list[curr_intr.user_id].push_back(idx);
			USER_BEHAV_CNT[curr_intr.user_id][curr_intr.behav] += 1;
			idx+=1;
		}

		// Populate Links

		ifstream link_file(link_path);

		while (getline(link_file,line)){
			// i, j (link from Interaction i -> Interaction j)
			vector<string> cols = split(line, '\t');

			int i,j;

			stringstream(cols[0]) >> i;
			stringstream(cols[1]) >> j;

			corpus[i].links.push_back(j);
			corpus[j].links_rev.push_back(i);
		}
	}

	void initialize(const char *corpus_path,const char *link_path)
	{
		cout<<"Initialization Started"<<endl;

		// Load the Behavior and Word Vocabulary
		readVocab(corpus_path);

		U = user_set.size();
		B = behav_set.size();

		user_intr_list.resize(U);

		USER_BEHAV_CNT.resize(U);

		for(int u = 0; u < U; ++u)
		{
			USER_BEHAV_CNT[u].resize(B,0);
		}

		// Load the Corpus
		readCorpus(corpus_path, link_path);

		cout<<"Corpus reading done"<<endl;

		// Determine the size of vocabulary, number of behaviors, number of Users and the size of the corpus
		V = vocab.size();
		C = corpus.size();

		T = 0;				// Set number of tables to 0 initially

		// Resize count matrices and arrays (ngu, nkb, nkw, ng, nk, intr_table, ntgk, ntg, n_links)

		ngu.resize(G);
		PHI_G_U_num.resize(G);
		PHI_G_U_denom.resize(G);

		for(int g = 0; g<G; ++g)
		{
			ngu[g].resize(U, 0);
			PHI_G_U_num[g].resize(U, 0.0);
		}

		nkb.resize(K);
		nkw.resize(K);
		// pkt.resize(K);

		PHI_K_B_num.resize(K);
		PHI_K_W_num.resize(K);

		PHI_K_W_denom.resize(K, 0.0);
		PHI_K_B_denom.resize(K, 0.0);


		for(int k = 0; k < K; ++k)
		{
			nkb[k].resize(B, 0);
			// pkt[k].resize(RESOLUTION, 0.0);
			PHI_K_B_num[k].resize(B, 0.0);
		}

		for(int k = 0; k < K; ++k)
		{
			nkw[k].resize(V, 0);
			PHI_K_W_num[k].resize(V, 0.0);
		}

		ng.resize(G, 0);

		nkbsum.resize(K, 0);
		nkwsum.resize(K, 0);

		user_table.resize(U, -1);

		user_group.resize(U, -1);
		intr_topic.resize(C, -1);

		intr_group.resize(C, -1);

		ntg.resize(G, 0);

		ngk.resize(G);
		ngksum.resize(G, 0);

		n_links.resize(G);

		pgkt.resize(G);

		double group_prior_prob = log(1.0/G);
		double group_topic_prior_prob = log(1.0/K);

		GROUP_PRIOR.resize(G, group_prior_prob);
		GROUP_G_K_PRIOR_NUM.resize(G);
		GROUP_G_K_PRIOR_DENOM.resize(G, 0.0);

		g_const_num.resize(G, 0.0);

		g_tables.resize(G);

		for(int g = 0; g < G; ++g)
		{
			ngk[g].resize(K, 0);
			n_links[g].resize(G, 0);

			GROUP_G_K_PRIOR_NUM[g].resize(K, group_topic_prior_prob);

			pgkt[g].resize(K);

			for(int k = 0; k < K; ++k)
			{
				pgkt[g][k].resize(RESOLUTION, 0.0);
			}
		}


		alpha_g.resize(G);
		beta_g.resize(G);

		for(int g = 0; g < G; ++g)
		{
			alpha_g[g].resize(K, 1.0);
			beta_g[g].resize(K, 1.0);
		}

		PY_PROB.resize(U, -1.0);

		// Modify Parameters
		alpha_gk = 10.0/K;
		alpha_gt = 10.0/G;

		alpha_b = 50.0/B;


		empty_tables.clear();

		double group_prob = 1.0/G;
		double topic_prob = 1.0/K;

		vector<double> p_g;
		vector<double> p_k;
		

		p_g.resize(G, group_prob);
		p_k.resize(K, topic_prob);


		// Initial Assignment of group and topic

		for(int u = 0; u < U ; u++)
		{
			int g = sample_from_prob(p_g);

			ngu[g][u] += 1;
			ng[g] += 1;


			user_group[u] = g;

			int num_interactions = user_intr_list[u].size();

			for(int i = 0; i < num_interactions;i++)
			{
				interaction curr_intr = corpus[user_intr_list[u][i]];

				intr_group[user_intr_list[u][i]] = g;

				int k = sample_from_prob(p_k);

				ngk[g][k] += 1;
				ngksum[g] += 1;

				intr_topic[user_intr_list[u][i]] = k;

				int b = curr_intr.behav;
				nkb[k][b] += 1;
				nkbsum[k] += 1;

				int W = curr_intr.text.size();

				for(int j = 0; j < W; ++j)
				{
					int w = curr_intr.text[j];
					nkw[k][w] += 1;
					nkwsum[k] += 1;
				}
			}
		}

		// Compute PHIs using initial cnts

		// PHI_G_U
		for(int g = 0; g < G; ++g)
		{
			for(int u = 0; u < U; ++u)
			{
				PHI_G_U_num[g][u] = log((ngu[g][u] + alpha_u));
			}

			PHI_G_U_denom[g] = log((ng[g] + (U * alpha_u)));
		}


		for(int k = 0; k < K; ++k)
		{

			for(int b = 0; b < B; ++b)
			{
				PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
			}

			PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));
		}

		for(int k = 0; k < K; ++k)
		{
			for(int v = 0; v < V; ++v)
			{
				PHI_K_W_num[k][v] = log((nkw[k][v] + alpha_w));
			}

			PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));
		}

	}


	void output_data_stats()
	{
		cout<<endl;

		cout<<"Number of Tables: "<<T<<endl;

		cout<<"Size of Corpus: "<<C<<endl;

		cout<<"Number of Users: "<<U<<endl;

		cout<<"Vocab Size: "<<V<<endl;

		cout<<"Number of Behaviors: "<<B<<endl;

	}


	void output_result(string destpath)
	{
		// Output Mappings

		// Vocab Mapping
		// string vocab_map_filename = "vocab-mapping.txt";

		ofstream vocab_map_file;
		vocab_map_file.open((destpath+vocab_map_filename).c_str(), ofstream::out);

		for(int j = 0; j < V; j++)
		{
			vocab_map_file<<j<<"\t"<<wordMapRev[j]<<endl;
		}

		vocab_map_file.close();


		// Behav Mapping
		// string behavior_map_filename = "behavior-mapping.txt";

		ofstream behav_map_file;
		behav_map_file.open((destpath+behavior_map_filename).c_str(), ofstream::out);

		for(int j = 0; j < B; ++j)
		{
			behav_map_file<<j<<"\t"<<behavMapRev[j]<<endl;
		}

		behav_map_file.close();


		// Group Priors
		// string group_prior_filename = "group-priors.txt";

		ofstream group_prior_file;
		group_prior_file.open((destpath+group_prior_filename).c_str(), ofstream::out);

		for(int g = 0; g < G; ++g)
		{
			double prob = (ntg[g] + alpha_gt) / ( (T - empty_tables.size())+ G * (alpha_gt));
			group_prior_file<<prob<<endl;
		}

		group_prior_file.close();


		string group_w_topic_filename = "group-topic-distribution.txt";

		ofstream group_w_topic_file;
		group_w_topic_file.open((destpath+group_w_topic_filename).c_str(), ofstream::out);

		for(int g = 0; g < G; ++g)
		{
			for(int k = 0; k < K; ++k)
			{
				double prob = ((ngk[g][k] + alpha_gk) / (ngksum[g] + K * (alpha_gk)));
				group_w_topic_file<<prob<<" ";
			}

			group_w_topic_file<<endl;
		}

		group_w_topic_file.close();


		// Group User Distribution
		// string group_user_filename = "group-user-distribution.txt";

		ofstream group_user_file;
		group_user_file.open((destpath+group_user_filename).c_str(), ofstream::out);

		for(int g = 0; g < G; ++g)
		{
			for(int u = 0; u < U; u++)
			{
				double prob = ((ngu[g][u] + alpha_u) / (ng[g] + (U * alpha_u)));
				group_user_file<<prob<<" ";
			}

			group_user_file<<endl;
		}

		group_user_file.close();



		// Output link prob from group g to group j
		// string link_prob_filename = "link-prob.txt";

		ofstream link_prob_file;
		link_prob_file.open((destpath+link_prob_filename).c_str(), ofstream::out);

		for(int g = 0; g < G; ++g)
		{
			for(int j = 0; j < G; j++)
			{

				double prob = ((n_links[g][j] + lambda_0) / (n_links[g][j] + n_links[j][g] + lambda_0 + lambda_1));
				link_prob_file<<prob<<" ";
			}

			link_prob_file<<endl;
		}

		link_prob_file.close();

		// Group Time Alpha
		// string group_time_alpha_filename = "group-time-alpha.txt";

		ofstream group_time_alpha_file;
		group_time_alpha_file.open(((destpath+group_time_alpha_filename)).c_str(), ofstream::out);

		for(int g = 0; g < G; ++g)
		{
			for(int k = 0; k < K; ++k)
			{
				group_time_alpha_file<<alpha_g[g][k]<<" ";
			}

			group_time_alpha_file<<endl;
		}

		group_time_alpha_file.close();


		// Group Time Beta
		// string group_time_beta_filename = "group-time-beta.txt";
		ofstream group_time_beta_file;
		group_time_beta_file.open(((destpath+group_time_beta_filename)).c_str(), ofstream::out);

		for(int g = 0; g < G; ++g)
		{
			for(int k = 0; k < K; ++k)
			{
				group_time_beta_file<<beta_g[g][k]<<" ";
			}

			group_time_beta_file<<endl;
		}

		group_time_beta_file.close();

		// Topic Word Distribution
		// string topic_word_filename = "topic-word-distribution.txt";

		ofstream topic_word_file;
		topic_word_file.open((destpath+topic_word_filename).c_str(), ofstream::out);

		for(int k = 0; k < K; ++k)
		{
			for(int v = 0; v < V; ++v)
			{
				double prob = ((nkw[k][v] + alpha_w)/(nkwsum[k] + (V * alpha_w)));

				topic_word_file<<prob<<" ";
			}

			topic_word_file<<endl;
		}

		topic_word_file.close();

		// Topic Behavior Distribution
		// string topic_behav_filename = "topic-behavior-distribution.txt";

		ofstream topic_behav_file;
		topic_behav_file.open((destpath+topic_behav_filename).c_str(), ofstream::out);

		for(int k = 0; k < K; ++k)
		{
			for(int b = 0; b < B; ++b)
			{
				double prob = ((nkb[k][b] + alpha_b) / (nkbsum[k] + (B * alpha_b)));
				topic_behav_file<<prob<<" ";
			}

			topic_behav_file<<endl;
		}

		topic_behav_file.close();


		// Table Seating File
		// string table_seating_filename = "table-assignment-status.txt";

		ofstream table_seating_file;
		table_seating_file.open((destpath+table_seating_filename).c_str(), ofstream::out);

		for(int t = 0; t < T; ++t)
		{
			table_seating_file<<n_table[t]<<"\t";
			table_seating_file<<table_group[t]<<"\t";

			for(int u = 0; u < U; ++u)
			{
				if(user_table[u] == t)
				{
					table_seating_file<<u<<", ";
				}
			}

			table_seating_file<<endl;

		}

		table_seating_file.close();

		// Top Words in each topic
		// string top_topic_words_filename = "top-topic-words.txt";
		ofstream top_topic_words_file;
		top_topic_words_file.open((destpath+top_topic_words_filename).c_str(), ofstream::out);

		
		map<int, string>::iterator it;

		if(num_top_words >=V)
		{
			num_top_words = V;
		}

		for (int k = 0; k < K; k++) {
			vector<pair<int, double> > words_probs;
			pair<int, double> word_prob;
			for (int w = 0; w < V; w++) {
				word_prob.first = w;
				word_prob.second = ((nkw[k][w] + alpha_w)/(nkwsum[k] + (V * alpha_w)));
				words_probs.push_back(word_prob);
			}

			// quick sort to sort word-topic probability
			quicksort(words_probs, 0, words_probs.size() - 1);

			top_topic_words_file<<"Topic "<<k<<":";

			for (int i = 0; i < num_top_words; i++) {
				it = wordMapRev.find(words_probs[i].first);
				if (it != wordMapRev.end()) {
					top_topic_words_file<<(it->second).c_str()<<"("<<words_probs[i].second<<")"<<"\t";
				}
			}

			top_topic_words_file<<endl;
		}
		top_topic_words_file.close();

		// Top Behavior in each Topic
		// string top_topic_behav_filename = "top-topic-behav.txt";
		ofstream top_topic_behav_file;
		top_topic_behav_file.open((destpath+top_topic_behav_filename).c_str(), ofstream::out);

		for (int k = 0; k < K; k++) {
			vector<pair<int, double> > words_probs;
			pair<int, double> word_prob;
			for (int b = 0; b < B; b++) {
				word_prob.first = b;
				word_prob.second = ((nkb[k][b] + alpha_b) / (nkbsum[k] + (B * alpha_b)));
				words_probs.push_back(word_prob);
			}

			// quick sort to sort word-topic probability
			quicksort(words_probs, 0, words_probs.size() - 1);

			top_topic_behav_file<<"Topic "<<k<<":";

			for (int i = 0; i < B; i++) {
				it = behavMapRev.find(words_probs[i].first);
				if (it != wordMapRev.end()) {
					top_topic_behav_file<<(it->second).c_str()<<"("<<words_probs[i].second<<")"<<"\t";
				}
			}

			top_topic_behav_file<<endl;
		}

		top_topic_behav_file.close();

		// Top Group Users

		ofstream top_group_user_file;
		top_group_user_file.open((destpath+top_group_user_filename).c_str(), ofstream::out);

		for (int g = 0; g < G; g++) {
			vector<pair<int, double> > words_probs;
			pair<int, double> word_prob;
			for (int u = 0; u < U; u++) {
				word_prob.first = u;
				word_prob.second = ((ngu[g][u] + alpha_u) / (ng[g] + (U * alpha_u)));
				words_probs.push_back(word_prob);
			}

			// quick sort to sort word-topic probability
			quicksort(words_probs, 0, words_probs.size() - 1);

			top_group_user_file<<"Group "<<g<<":";

			for (int i = 0; i < num_top_users; i++) {
				top_group_user_file<<words_probs[i].first<<"("<<words_probs[i].second<<")"<<"\t";
			}

			top_group_user_file<<endl;
		}

		top_group_user_file.close();


		// Print g_k_tables

		ofstream g_k_tables_file;
		g_k_tables_file.open((destpath + g_k_tables_filename).c_str(), ofstream::out);

		for(int g = 0; g < G; ++g)
		{
			vector<int> table_li = g_tables[g];
			int num_tables = table_li.size();
			g_k_tables_file<<g<<"\t";

			for(int j = 0; j < num_tables; ++j)
			{
				int t = table_li[j];
				g_k_tables_file<<t<<",";
			}
			g_k_tables_file<<endl;
		}

		g_k_tables_file.close();
	}
};
