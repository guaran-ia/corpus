# Integrated Curation Pipeline

This directory contains the reusable code for the curation Pipeline used to create versions of KuatIA, our Guarani dataset. 

This document provides an explanation of the classes. To learn how to use the code, we recommend you check out our [tutorial](examples/tutorial.ipynb).

## Documents

A [`Document`](utils/document.py) is a data class representing a single text file. Each Document consists of its `text`, an `id`, and optional additional `metadata`. Documents are mostly for internal usage, and a Pipeline user will typically not deal with Documents.

Currently, the pipeline supports Documents from `.jsonl` files, where each line represents a single Document.

## Pipeline Steps

Pipeline Steps are the basic components of the Pipeline. All implemented steps are child classes of the [`PipelineStep`](pipeline_steps/pipeline_step.py) class. 

While the constructors of each step are different, all Pipeline Steps:

- Implement a `run()` method, which executes the step. This method takes two arguments:
    - `input_directory`: a path to the directory with the input `.jsonl` files that will be processed by the step.
    - `output directory:` a path to the directory where the output files of the step will be written.

- Generate a report (see [`Step Report`](#step-report)) in a `report.json` file inside the specified output directory, containing information about the step run.

- Return the directory where the useful/clean output (e.g. the deduplicated Documents, or the Documents that pass a filter) is stored.

### Step Report

The StepReport is a class that contains information about the implementation of a particular Pipeline Step. It consists of the following fields:

- `step_name`: the name of the step.

- `input_directory`: the **relative** directory where the input files of the step can be found.

- `output_directory`: the **relative** directory where the output files (i.e. those containing the clean/useful) of the step can be found.

- `step_stats`: stats and information that are specific to the step.

- `input_documents`: the number of Documents before the step.

- `output_documents`: the number of clean/useful Documents after the step, it may or may not differ from `input_documents`, according to the step.

- `input_files`: the number of files before the step.

- `file_stats`: a file-by-file breakdown of the step.

### Currently Implemented Steps

#### URL Deduplication

Removes duplicated Documents according to their URLs, leaving a single Document per URL. 

##### Inputs

To be instantiated it needs:

- `duplicate_ids_path`: the path to a `.json` output of running [**URL deduplication**](../deduplication/url_deduplication.py) as implemented in [`src/pipeline/deduplication`](../deduplication), on the intended source directory.

- `ignore_files_path` *(optional)*: the path to a `.json` output of [**checking for intracorpus URLs**](../deduplication/check_intracorpus_urls.py) as implemented in [`src/pipeline/deduplication`](../deduplication), on the intended source directory.

##### Outputs

The `run()` function creates the following two subdirectories in the specified `output_directory`:

- `keep/`: where the deduplicated Documents are written in files mirroring their original source files. **This is the directory returned by the function**
- `remove/`: where the duplicate Documents are written in files mirroring their original source files.


#### MinHash Deduplication

Removes duplicated Documents according to their MinHash signatures, leaving a single Document per cluster (set of Documents with similar signatures).

##### Inputs

To be instantiated it needs:

- `duplicate_ids_path`: the path to a `.json` output of running [**MinHash deduplication**](../deduplication/run_minhash.py) as implemented in [`src/pipeline/deduplication`](../deduplication), on the intended source directory.

##### Outputs

The `run()` function creates the following two subdirectories in the specified `output_directory`:

- `keep/`: where the deduplicated Documents are written in files mirroring their original source files. **This is the directory returned by the function**
- `remove/`: where the duplicate Documents are written in files mirroring their original source files.

#### PII Censoring

Censors spans in Documents identified as containing Personal and Identifiable Information (PII) according to a chosen censoring method.

##### Inputs

To be instantiated, it needs:

- `censoring_method` *(optional)*: the censoring method to use. It must be one of the following options:
    - `"tag"` replaces the PII span with a tag in curly braces indicating the type of PII (e.g. {email}). **This is the default option.**
    - `"stars"` replaces the PII span with a string of asterisks ('*') of the same length as the span.
    - `"replace"` replaces the PII span with a randomised text of the same type as the PII span (e.g. a real e-mail address gets replaced by "johndoe@domain.com")
- `overlap_handle` *(optional)*: how to handle ovelapping spans, it contains two fields, **which must both be specified if specifying the overlap handle**:
    - `"merge"`: a boolean *(`True` or `False`)* value specifying whether to merge overlapping spans of the same type.
    - `"priority"`: defines how to handle overlaps of spans of different types. Must be one of the following option:
        - `"left"`: prioretises the leftmost span.
        - `"right"`: prioretises the rightmost span.
        - `"large"`: prioretises the largest span.
        - `"small"`: prioretises the smallest span.
    
> [!WARNING] 
> For the `run()` function of this Step, it is imperative that the `input_directory` files be passed through the [**PII identification script**](../pii/run_pii_metrics.py), as specified in the [**PII directory**](../pii/), so that the input Documents may have the necessary metadata fields to execute this step. 

##### Outputs

The `run()` function creates a single `censored/` directory in the specified `output_directory`. This directory contains all the censored Documents in files mirroring their original source files and is returned by the function.

#### Heuristic Filter

Removes Documents according to a filtering condition using the Documents's metadata.

##### Inputs

To be instantiated, it needs:

- `filtering_condition`: the condition used to filter out the Documents. It is a string representing a `python` boolean condition, where the variables are the Documents' `metadata` fields. The syntax is as follows:
    - The Documents' metadata are used as fields. E.g.: `"language_score < 0.9"`.
    - Global percentiles of fields can be specified as `p<N>_<field>`, with *`N`* being the percentile (use an underscore '_' as a decimal point), and *`field`* being the name of the metadata field. E.g.: `"language_score < p99_language_score"` or `"num_words_split < p1_5_num_words_split"`.
    - Conditions can be combined using parentheses and standard `python` logical operators. E.g.: `"(language_score < 0.9) and (num_words_split < p1_5_num_words_split)"`

> [!WARNING] 
> For this type of Step to execute correctly, all Documents must contain any specified fields, otherwise the condition cannot be evaluated.

##### Outputs

The `run()` function creates the following two subdirectories in the specified `output_directory`:

- `keep/`: where the Documents not removed by the `filtering_condition` are written in files mirroring their original source files. **This is the directory returned by the function**
- `remove/`: where the Documents removed by the `filtering_condition` are written in files mirroring their original source files.

## Pipeline 

A [`Pipeline`](pipeline.py) is a combination of one or more Pipeline Steps that are excecuted sequentially in a single run, with each Step taking the previous Step's output as an input. 

A Pipeline instance receives as arguments:

- `steps`: a list of the sequence of steps to be executed in the order in which they appear in the list. The steps may be instances of a PipelineStep child class, or a dictionary detailing the step type and configs (see our tutorial for guidance).

- `input_directory`: a path to the directory with the input `.jsonl` files that will be processed by the Pipeline, these will be the input of the first Step.

- `output_directory`:

- `start`: an integer detailing the start of the numeration of steps, it defaults to 1.

## Experiments

An Experiment is a way to evaluate a Pipeline to determine the quality

## Dataset Version

A dataset version is a collection of files containing