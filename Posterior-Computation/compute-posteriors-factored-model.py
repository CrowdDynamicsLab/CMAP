import codecs
import os
import math
import numpy as np 
import scipy.special
import operator
import datetime

import argparse


def load_phi_w(phi_w_path):
	PHI_W = []

	file = codecs.open(phi_w_path, 'r', 'utf-8')

	for row in file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		PHI_W.append(curr_li)

	file.close()

	return PHI_W


def load_phi_b(phi_b_path):
	PHI_B = []

	file = codecs.open(phi_b_path, 'r', 'utf-8')

	for row in file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		PHI_B.append(curr_li)

	file.close()

	return PHI_B

def load_time_map(mapping_path):

	TIME_MAP = {}

	mapping_file = codecs.open(mapping_path, 'r', 'utf-8')

	idx = 0

	for row in mapping_file:
		# UserId, PostId, Behav, TimeStamp

		s = row.strip().split('\t')
		
		struct_time = datetime.datetime.strptime(s[3], "%Y-%m-%d %H:%M:%S")
		# Id = int(s[1])
		# actual_ts = int(s[3].strip())

		TIME_MAP[idx] = struct_time

		idx+=1

	return TIME_MAP

def load_link_probs(link_prob_path):
	LINK_PROB = []

	link_prob_file = codecs.open(link_prob_path, 'r', 'utf-8')

	for row in link_prob_file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		LINK_PROB.append(curr_li)

	link_prob_file.close()

	return LINK_PROB



def load_alpha_beta_k():
	ALPHA_K = []

	alpha_file = codecs.open(alpha_k_path, 'r', 'utf-8')

	for row in alpha_file:
		ALPHA_K.append(float(row.strip()))

	alpha_file.close()


	BETA_K = []

	beta_file = codecs.open(beta_k_path, 'r', 'utf-8')

	for row in beta_file:
		BETA_K.append(float(row.strip()))

	beta_file.close()

	return ALPHA_K, BETA_K

def load_alpha_beta_g(alpha_g_path, beta_g_path):
	ALPHA_G = []

	alpha_file = codecs.open(alpha_g_path, 'r', 'utf-8')

	for row in alpha_file:
		curr_alpha = []
		s = row.strip().split(' ')
		for elem in s:
			curr_alpha.append(float(elem))

		ALPHA_G.append(curr_alpha)

	alpha_file.close()


	BETA_G = []

	beta_file = codecs.open(beta_g_path, 'r', 'utf-8')

	for row in beta_file:
		curr_beta = []

		s = row.strip().split(' ')

		for elem in s:
			curr_beta.append(float(elem))
		BETA_G.append(curr_beta)

	beta_file.close()

	return ALPHA_G, BETA_G


def load_group_user_distr(group_user_distr_path):
	GROUP_USER = []

	group_user_distr_file = codecs.open(group_user_distr_path, 'r', 'utf-8')

	for row in group_user_distr_file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		GROUP_USER.append(curr_li)

	group_user_distr_file.close()

	return GROUP_USER

def load_group_prior(group_prior_path):
	GROUP_PRIOR = []

	group_prior_file = codecs.open(group_prior_path, 'r', 'utf-8')

	for row in group_prior_file:
		GROUP_PRIOR.append(float(row.strip()))

	group_prior_file.close()

	return GROUP_PRIOR

def load_group_topic_distr(group_b_topic_distr_path, group_w_topic_distr_path):
	GROUP_B_TOPIC = []

	GROUP_W_TOPIC = []

	group_topic_distr_file = codecs.open(group_b_topic_distr_path, 'r', 'utf-8')

	for row in group_topic_distr_file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		GROUP_B_TOPIC.append(curr_li)


	group_topic_distr_file.close()


	group_topic_distr_file = codecs.open(group_w_topic_distr_path, 'r', 'utf-8')

	for row in group_topic_distr_file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		GROUP_W_TOPIC.append(curr_li)


	group_topic_distr_file.close()

	return GROUP_B_TOPIC, GROUP_W_TOPIC


def load_word_idx(word_idx_path):
	WORD_IDX = {}

	file = codecs.open(word_idx_path, 'r', 'utf-8')
	idx = 0
	for row in file:
		# idx+=1
		# print(idx)
		try:
			s = row.strip().split('\t')
			WORD_IDX[s[1]] = int(s[0])
		except:
			print("Error: ",row.strip())

	file.close()

	return WORD_IDX



def load_behav_idx(behav_idx_path):
	BEHAV_IDX = {}

	file = codecs.open(behav_idx_path, 'r', 'utf-8')

	for row in file:
		s = row.strip().split('\t')
		BEHAV_IDX[s[1]] = int(s[0])

	file.close()

	return BEHAV_IDX


def load_user_map(user_map_path):
	USER_MAP = {}

	file = codecs.open(user_map_path, 'r', 'utf-8')

	for row in file:
		s = row.strip().split('\t')
		USER_MAP[int(s[1])] = s[0]

	file.close()

	return USER_MAP


def load_user_group_topic(table_assign_path, discount):
	USER_GROUP = {}
	USER_TOPIC = {}

	USER_TABLE = {}

	PY_TERM = [[0.0]*K_b]*G

	# N_g = [0]*G
	# N_k = [0]*K

	C = 0

	table_idx = 0

	table_assign_file = codecs.open(table_assign_path, 'r', 'utf-8')

	for row in table_assign_file:
		# Num_intr, Group, Topic, (Interactions separated by ,)

		s = row.strip().split('\t')

		if(int(s[0]) == 0):
			table_idx+=1
			continue

		group = int(s[1])
		topic = int(s[2])

		intr_list = s[3].strip().split(",")

		for elem in intr_list:
			if elem.strip() == '':
				continue
			user = int(elem.strip())
			USER_GROUP[user] = group
			USER_TOPIC[user] = topic

			USER_TABLE[user] = table_idx
			# N_g[group]+=1
			# N_k[topic]+=1

			C+=1

		PY_TERM[group][topic] += (int(s[0]) - discount)

		table_idx+=1

	table_assign_file.close()

	return USER_GROUP, USER_TOPIC, USER_TABLE, PY_TERM


def load_post_ids(mapping_path):

	POST_IDs = {}

	mapping_file = codecs.open(mapping_path, 'r', 'utf-8')

	idx = 0

	for row in mapping_file:
		# UserId, Postid, Behav, CreationDate

		s = row.strip().split('\t')
		
		Id = int(s[1])

		POST_IDs[idx] = Id

		idx+=1

	return POST_IDs

def compute_time_prob(alpha, beta, t):
	prob = (1.0*(math.pow(t, alpha - 1))*(math.pow(1-t, beta - 1)))/(scipy.special.beta(alpha, beta))

	return prob

def compute_posteriors(INTR_GROUP, INTR_TOPIC, USER_MAP, WORD_IDX, BEHAV_IDX, PHI_W, PHI_B, GROUP_USER, ALPHA_G, BETA_G, ALPHA_K, BETA_K, GROUP_PRIOR, GROUP_B_TOPIC_PRIOR, GROUP_W_TOPIC_PRIOR, POST_IDs, LINKS_i_j, LINK_PROB, USER_TABLE, PY_TERM, TIME_MAP, model, dataset, discount, intr_path):

	DICT_USER_POSTERIORS = {}

	group_posteriors = []

	intr_file = codecs.open(intr_path, 'r', 'utf-8')

	idx = 0

	for row in intr_file:
		# Text, u, b, ts
		s = row.split('\t')

		# if s[1].startswith("TEMP_USER"):
		# 	continue

		text = s[0].strip()
		u = int(s[1].strip())
		b = s[2].strip()
		ts = float(s[3].strip())

		b = BEHAV_IDX[b]

		text_li = text.strip().split(' ')

		curr_posterior_g = []
		curr_posterior_k = []


		for g in range(G):
			prob_g = 0.0

			for k_w in range(K_w):
				prob_k = 1.0

				for word in text_li:
					if word == '':
						continue
					w = WORD_IDX[word]

					prob_k = prob_k * PHI_W[k_w][w]

				prob_k = prob_k * compute_time_prob(ALPHA_G[g][k_w], BETA_G[g][k_w], ts)

				prob_k *= GROUP_W_TOPIC_PRIOR[g][k_w]

				prob_g+=prob_k


			curr_posterior_g.append(prob_g)

		group_posteriors.append(curr_posterior_g)

		idx+=1

		if idx%1000 == 0:
			print(model, dataset, discount, idx)

	intr_file.close()

	print("Without link prob loaded")


	intr_file = codecs.open(intr_path, 'r', 'utf-8')

	idx = 0

	for row in intr_file:
		# Text, u, b, ts
		s = row.split('\t')

		# if s[1].startswith("TEMP_USER"):
		# 	continue

		text = s[0].strip()
		u = int(s[1].strip())

		if u == -1:
			idx+=1
			continue
		b = s[2].strip()
		ts = float(s[3].strip())

		b = BEHAV_IDX[b]

		text_li = text.strip().split(' ')


		curr_posterior_g_k_b = []

		for g in range(G):
			for k_b in range(K_b):
				prob = 1.0

				prob *= GROUP_PRIOR[g]

				prob *= PY_TERM[g][k_b]

				prob *= PHI_B[k_b][b]

				prob_k_w = 0.0

				for k_w in range(K_w):
					prob_k = 1.0

					for word in text_li:
						if word == '':
							continue
						w = WORD_IDX[word]

						prob_k = prob_k * PHI_W[k_w][w]

					prob_k = prob_k * compute_time_prob(ALPHA_G[g][k_w], BETA_G[g][k_w], ts)

					prob_k *= GROUP_W_TOPIC_PRIOR[g][k_w]

					prob_k_w+=prob_k

				prob *= prob_k_w

				if idx in LINKS_i_j:
					curr_links = LINKS_i_j[idx]

					for j in curr_links:
						posterior_j = group_posteriors[j]
						max_index, max_value = max(enumerate(posterior_j), key=operator.itemgetter(1))

						prob = prob * LINK_PROB[g][max_index]
				curr_posterior_g_k_b.append(prob)

		curr_posterior_g = []

		li_idx = 0
		for g in range(G):
			prob_g = 0.0
			for k in range(K_b):
				prob_g += curr_posterior_g_k_b[li_idx]

				li_idx+=1

			curr_posterior_g.append(prob_g)

		curr_posterior_k = [0.0]*K_b

		li_idx = 0
		for g in range(G):
			for k in range(K_b):
				curr_posterior_k[k] += curr_posterior_g_k_b[li_idx]
				li_idx+=1


		sum_li = np.sum(curr_posterior_g)
		if sum_li == 0.0:
			sum_li = np.sum([1.0])
		curr_posterior_g = curr_posterior_g/sum_li
		curr_posterior_g = curr_posterior_g.tolist()

		sum_li = np.sum(curr_posterior_k)
		if sum_li == 0.0:
			sum_li = np.sum([1.0])
		curr_posterior_k = curr_posterior_k/sum_li
		curr_posterior_k = curr_posterior_k.tolist()


		b = s[2].strip()

		# actual_ts = TIME_MAP[idx]

		# actual_user = USER_MAP[u]

		actual_user = u

		if actual_user not in DICT_USER_POSTERIORS:
			DICT_USER_POSTERIORS[actual_user] = []
		# g = INTR_GROUP[idx]

		DICT_USER_POSTERIORS[actual_user].append(curr_posterior_g+curr_posterior_k)

		idx+=1

		if idx%1000==0:
			print(model, dataset, discount, idx)


	intr_file.close()

	return DICT_USER_POSTERIORS


def load_links(links_path):

	LINKS_i_j = {}

	links_file = codecs.open(links_path, 'r', 'utf-8')

	for row in links_file:
		# Interaction i -> Interaction j
		s = row.strip().split('\t')

		i = int(s[0])
		j = int(s[1])

		if i not in LINKS_i_j:
			LINKS_i_j[i] = []

		LINKS_i_j[i].append(j)

	links_file.close()

	return LINKS_i_j


def generate_posteriors(basepath, intr_path, links_path, discount):
	# INTR_GROUP, INTR_TOPIC, USER_MAP, WORD_IDX, BEHAV_IDX, PHI_W, PHI_B, GROUP_USER, ALPHA_G, BETA_G, ALPHA_K, BETA_K, GROUP_PRIOR, GROUP_TOPIC_PRIOR

	# basepath = "../Output/" +model+"_"+str(K_b)+"_"+str(G)+"_"+dataset+"_"+str(discount)+"00000/"

	phi_w_path = basepath+"topic-word-distribution.txt"
	phi_b_path = basepath+"topic-behavior-distribution.txt"
	alpha_k_path = basepath+"topic-time-alpha.txt"
	beta_k_path = basepath+"topic-time-beta.txt"
	alpha_g_path = basepath+"group-time-alpha.txt"
	beta_g_path = basepath+"group-time-beta.txt"
	group_user_distr_path = basepath+"group-user-distribution.txt"
	group_prior_path = basepath+"group-priors.txt"
	group_b_topic_distr_path = basepath+"group-b-topic-distribution.txt"
	group_w_topic_distr_path = basepath+"group-w-topic-distribution.txt"
	word_idx_path = basepath+"vocab-mapping.txt"
	behav_idx_path = basepath+"behavior-mapping.txt"
	table_assign_path = basepath+"table-assignment-status.txt"

	# user_map_path = "../Data/"+dataset+"-user-map.txt"
	# intr_path = "../Data/"+dataset+"_pre_processed.txt"
	# mapping_path = "../Data/"+dataset+"_map.txt"
	# links_path = "../Data/"+dataset+"_links.txt"

	posterior_path = basepath+"posteriors-user-interactions.txt"

	link_prob_path = basepath+"link-prob.txt"



	USER_MAP = load_user_map(user_map_path)
	WORD_IDX = load_word_idx(word_idx_path)
	BEHAV_IDX = load_behav_idx(behav_idx_path)
	PHI_W = load_phi_w(phi_w_path)
	PHI_B = load_phi_b(phi_b_path)
	GROUP_USER = load_group_user_distr(group_user_distr_path)
	ALPHA_G, BETA_G = load_alpha_beta_g(alpha_g_path, beta_g_path)
	# ALPHA_K, BETA_K = load_alpha_beta_k()

	ALPHA_K = []
	BETA_K = []

	GROUP_PRIOR = load_group_prior(group_prior_path)
	GROUP_B_TOPIC_PRIOR, GROUP_W_TOPIC_PRIOR = load_group_topic_distr(group_b_topic_distr_path, group_w_topic_distr_path)

	# INTR_GROUP, INTR_TOPIC = load_intr_group_topic()

	USER_GROUP, USER_TOPIC, USER_TABLE, PY_TERM = load_user_group_topic(table_assign_path, discount)

	# USER_GROUP = {}
	# USER_TOPIC = {}
	# POST_IDs = load_post_ids(mapping_path)
	POST_IDs = {}

	LINKS_i_j = load_links(links_path)
	LINK_PROB = load_link_probs(link_prob_path)
	# LINKS_i_j = {}
	# LINK_PROB = []

	# TIME_MAP = load_time_map(mapping_path)
	TIME_MAP = {}

	print("Loading Done")

	DICT_USER_POSTERIORS = compute_posteriors(USER_GROUP, USER_TOPIC, USER_MAP, WORD_IDX, BEHAV_IDX, PHI_W, PHI_B, GROUP_USER, ALPHA_G, BETA_G, ALPHA_K, BETA_K, GROUP_PRIOR, GROUP_B_TOPIC_PRIOR, GROUP_W_TOPIC_PRIOR, POST_IDs, LINKS_i_j, LINK_PROB, USER_TABLE, PY_TERM, TIME_MAP, model, dataset, discount, intr_path)

	print("User Posterior Dict: ", len(DICT_USER_POSTERIORS))

	posterior_file = codecs.open(posterior_path, 'w', 'utf-8')

	for user in DICT_USER_POSTERIORS:
		print(str(user)+'\t'+str(DICT_USER_POSTERIORS[user]), file = posterior_file)

	posterior_file.close()

	print('Posteriors Saved to '+ posterior_path)


parser = argparse.ArgumentParser("Posterior_Factored")
parser.add_argument("--output_path", help="Path to CMAP output files")
parser.add_argument("--corpus_path", help="Path to pre_processed file")
parser.add_argument("--links_path", help="Path to links file")
parser.add_argument("--discount", help="Value of Discount Parameter", default = "0.5")
parser.add_argument("--K_b", help="Number of Behavior Topics", default = "5")
parser.add_argument("--K_w", help="Number of Text Topics", default = "20")
parser.add_argument("--G", help="Number of Profiles", default = "20")


args = parser.parse_args()

K_b = int(args.K_b)
K_w = int(args.K_w)
G = int(args.G)


discount = float(args.discount)

basepath = args.output_path
intr_path = args.corpus_path
links_path = args.links_path


generate_posteriors(basepath, intr_path, links_path, discount)
