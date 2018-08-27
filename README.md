# CRP-based Multifacet Activity Profiling Model (CMAP)

CMAP is a compact, generative framework for participant behavior representation on social learning platforms. It jointly models the temporal evolution of content, behavior and inter-participant links in a unified latent framework, integrated with the Chinese Restaurant Process (Pitman-Yor) to effectively deal with behavior skew and data sparsity. 

This model was proposed in the following paper:

Adit Krishnan, Ashish Sharma and Hari Sundaram, "Learning Fine-grained Latent Representations of Online User Behavior in the Presence of Skew and Sparsity", 27th ACM International Conference on Information and Knowledge Management (CIKM'18)

## Getting Started

These instructions will get you a copy of the model up and running on your local machine.

### Platforms Supported

- Unix-Like

### Prerequisites

Please ensure that the following dependencies are installed: 
- g++ (>=4.8.0)


### Development Setup

Run the following command for compiling the project

```
$ make
```

## Variations of Model Supported

This repository supports the following variations of CMAP:

- **Unified** - Modelling words and actions jointly using same set of topics.
- **Factored** - Modelling words and actions using different set of topics.

Both Unified and Factored model have a hierarchical counterparts where we introduce hierarchy over set of topics. In total, the repository supports 4 different variations of CMAP. All the 4 variations can either be run with parallelization (using threads, if multiple cores are available) or without parallelization (normal linear execution of code). 

Please refer the paper for details. 

## Input File Format

For a paricular dataset, the model requires 2 input files:

- **<dataset>_pre_processed.txt**: Each row of this file corresponds to one data point in your dataset and has 4 columns - Text, UserId, Behaviour and Timestamp (all tab separated).  The columns are described below:
    - **Text**: The text in your data point. Pre-process the text for efficient use.
    - **UserId**: A user index between 0 to num_users-1 corresponding to the user of the data point.
    - **Behaviour**: The action observed in the data point. Eg, questioning, answering, commenting, etc. for Stack-Exchanges
    - **Timestamp**: The normalized value of time of data point. The value must be between 0.01 to 0.99 and should be truncated to 2 decimal places. 
- **<dataset>_links.txt**: This file is optional and can be provided if you have social interaction information as part of your dataset. Each row corresponds to a link from a data point i to data point j and has 2 colums - i and j (tab-separated). The data points are zero-indexed and indexing is defined from the <dataset>_pre_processed file.

Both the files should be put inside the **Data** folder.

## Running the Model

The model can be executed using the following command.

```
$ ./CMAP <options>
```

where possible options include:

```
--dataset <name>:         Name of the dataset to use (required)
--model <type>:           Model-Type. 0 for unified, 1 for factored (default 0)
--hr:                     Use the hierarachical version of the model
--thread <num_thread>:    Use the threaded version of the model. Specify number of threads to use
--G <num_groups>:         Specify the value of number of groups (default: 20)
--K <num_topics>:         Specify number of topics (Use only for unified model) (default: 20)
--K_w <num_text_topics>:  Specify number of text topics (Use only for factored model) (default: 20)
--K_b <num_behav_topics>: Specify number of behavior topics (Use only for factored model) (default: 5)
--scale <s>:              Specify the value of scale parameter (default: 1.5)
--discount <d>:           Specify the value of dicount parameter (default: 0.5)
--iter <num_iter>:        Specify the number of iterations to run (default: 500)
--help:                   Print help
```

Providing dataset name is mandatory. Please refer to the paper for optimal values of these parameters.

## Sample Run
A sample dataset named biology is present in the Data folder. For running the hierarchical variation of the unified model with G = 20 and K = 10 for 100 iterations, execute the following command:
```
$ ./CMAP --dataset biology --model 0 --hr --G 20 --K 10 --iter 100
```

## Output
For each full run of CMAP, a folder named **<CMAP_Model_Type>\_<K>\_<G>\_<dataset>** will be created in the Output Folder. The files of the output folder are described below:

- **vocab-mapping.txt:** Words to indices Mapping.
- **behavior-mapping.txt:** Actions to indices Mapping. 
- **group-priors.txt:** Prior probability of groups.
- **group-topic-distribution.txt:** Distribution of groups over topics.
- **group-user-distribution.txt:** Distribution of groups over users.
- **link-prob.txt:** Probability of having a link from group i to group j.
- **group-time-alpha.txt:** Group-Time Alpha values.
- **group-time-beta.txt:** Group-Time Beta values.
- **topic-word-distribution.txt:** Topic to Word Distribution.
- **topic-behavior-distribution.txt:** Topic to Behavior Distribution.
- **table-assignment-status.txt:** Status of Data points seating.
- **top-topic-words.txt:** Top 20 words in a topic.
- **top-topic-behav.txt:** Top behaviors in a topic.
- **top-group-users.txt:** Top users in a group.

## References
