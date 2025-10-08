#!/usr/bin/env python3
"""Sequence aligner script"""

import configparser
import os

import psycopg2

from . import create_web_app, get_config, parse_files, run_vsa
from .passage_classifier import classify_passages
from .sequence_alignment import (
    Ngrams,
    banality_auto_detect,
    banality_llm_post_eval,
    merge_alignments,
    phrase_matcher,
    zero_shot_banality_detection,
)


def delete_database(dbname: str) -> None:
    global_config = configparser.ConfigParser()
    global_config.read("/etc/text-pair/global_settings.ini")
    conn = psycopg2.connect(
        user=global_config["DATABASE"]["database_user"],
        password=global_config["DATABASE"]["database_password"],
        database=global_config["DATABASE"]["database_name"],
    )
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        print(f"Dropping table {dbname}...", end="")
        cursor.execute(f"DROP TABLE IF EXISTS {dbname}")
        print("done")
        print(f"Dropping table {dbname}__ordered...", end="")
        cursor.execute(f"DROP TABLE IF EXISTS {dbname}__ordered")
        print("done")
        print(f"Dropping table {dbname}__groups...", end="")
        cursor.execute(f"DROP TABLE IF EXISTS {dbname}___groups")
        print("done")
        print(f"Deleting {dbname} web app directory...", end="")
        os.system(f"rm -rf {global_config['WEB_APP']['web_app_path']}/{dbname}")
        print("done")

        print(f"\nDeletion of database {dbname} complete.")
    except Exception as e:
        print(e)
    finally:
        conn.close()


def get_count(path: str) -> int:
    """Get count from count file"""
    if os.path.exists(path):
        with open(path, encoding="utf8") as input_file:
            count = int(input_file.read().strip())
    else:
        count = 0  # TODO: handle this case by counting matches in results file
    return count


def update_count(count: int, to_remove: int, path: str) -> int:
    """Update count file"""
    path = os.path.join(path, "results/count.txt")
    count -= to_remove
    with open(path, "w", encoding="utf8") as output_file:
        output_file.write(str(count))
    return count


async def run_alignment(params):
    """Main function to start sequence alignment"""
    if params.only_align is False:
        if params.text_parsing["parse_source_files"] is True:
            print("\n### Parsing source files ###")
            parse_files(
                params.paths["source"]["input_files"],
                params.text_parsing["source_file_type"],
                params.paths["source"]["input_source_metadata"],
                params.paths["source"]["parse_output"],
                params.text_parsing["source_words_to_keep"],
                params.preprocessing_params["source"]["text_object_type"],
                params.preprocessing_params["source"]["lowercase"],
                params.workers,
                params.debug,
            )
        print("\n### Generating source ngrams ###")
        ngrams = Ngrams(debug=params.debug, **params.preprocessing_params["source"])
        ngrams.generate(
            params.paths["source"]["input_files_for_ngrams"],
            params.paths["source"]["ngram_output_path"],
            params.workers,
        )
        if params.paths["target"]:
            if params.text_parsing["parse_target_files"] is True:
                print("\n### Parsing target files ###")
                parse_files(
                    params.paths["target"]["input_files"],
                    params.text_parsing["target_file_type"],
                    params.paths["target"]["input_target_metadata"],
                    params.paths["target"]["parse_output"],
                    params.text_parsing["target_words_to_keep"],
                    params.preprocessing_params["target"]["text_object_type"],
                    params.preprocessing_params["target"]["lowercase"],
                    params.workers,
                    params.debug,
                )
            print("\n### Generating target ngrams ###")
            ngrams = Ngrams(debug=params.debug, **params.preprocessing_params["target"])
            ngrams.generate(
                params.paths["target"]["input_files_for_ngrams"],
                params.paths["target"]["ngram_output_path"],
                params.workers,
            )
    print("\n### Starting sequence alignment ###")
    if params.paths["target"]["ngram_output_path"] == "":  # if path not defined make target like source
        params.paths["target"]["ngram_output_path"] = params.paths["source"]["ngram_output_path"]
    result_batch_path = os.path.join(params.output_path, "results/result_batches")
    if os.path.exists(result_batch_path):
        os.system(f"rm -rf {result_batch_path}")
    command = f"""compareNgrams \
                --output_path={params.output_path}/results \
                --threads={params.workers} \
                --source_files={params.paths["source"]["ngram_output_path"]}/ngrams \
                --target_files={params.paths["target"]["ngram_output_path"]}/ngrams \
                --source_metadata={params.paths["source"]["metadata_path"]} \
                --target_metadata={params.paths["target"]["metadata_path"]} \
                --sort_by={params.matching_params["sort_by"]} \
                --source_batch={params.matching_params["source_batch"]} \
                --target_batch={params.matching_params["target_batch"]} \
                --matching_window_size={params.matching_params["matching_window_size"]} \
                --max_gap={params.matching_params["max_gap"]} \
                --flex_gap={params.matching_params["flex_gap"]} \
                --minimum_matching_ngrams={params.matching_params["minimum_matching_ngrams"]} \
                --minimum_matching_ngrams_in_window={params.matching_params["minimum_matching_ngrams_in_window"]} \
                --minimum_matching_ngrams_in_docs={params.matching_params["minimum_matching_ngrams_in_docs"]} \
                --context_size={params.matching_params["context_size"]} \
                --duplicate_threshold={params.matching_params["duplicate_threshold"]} \
                --merge_passages_on_byte_distance={params.matching_params["merge_passages_on_byte_distance"]} \
                --merge_passages_on_ngram_distance={params.matching_params["merge_passages_on_ngram_distance"]} \
                --passage_distance_multiplier={params.matching_params["passage_distance_multiplier"]} \
                --debug={str(params.debug).lower()} \
                --ngram_index={params.matching_params["ngram_index"]}"""
    results_file = f"{params.output_path}/results/alignments.jsonl.lz4"
    if os.path.exists(results_file):
        os.system(f"rm -rf {results_file}")
    if params.debug:
        print(f"Running alignment with following arguments:\n{' '.join(command.split())}")
    os.system(command)
    if len(os.listdir(result_batch_path)) == 1:
        filename = os.listdir(result_batch_path)[0]
        os.system(f"mv {result_batch_path}/{filename} {results_file} && rm -rf {result_batch_path}")
    else:
        print("Merging alignments into one file (this may take a while)... ", end="", flush=True)
        merge_command = f"find {result_batch_path} -type f | sort -V | xargs lz4cat --rm | lz4 -q > {results_file}; rm -rf {result_batch_path}"
        os.system(merge_command)
        print("done.")
    count = get_count(os.path.join(params.output_path, "results/count.txt"))

    # Postprocessing steps
    if any([params.matching_params["phrase_filter"], params.matching_params["banality_auto_detection"], params.matching_params["banality_llm_eval"]]):
        print(f"\n### Postprocessing {count} pairwise alignments ###")
        if params.matching_params["phrase_filter"]:
            filtered_passages = phrase_matcher(results_file, params.matching_params["phrase_filter"], count)
            print(f"{filtered_passages} pairwise alignments have been filtered based on the phrase filter provided.")
            count = update_count(count, filtered_passages, params.output_path)
            print(f"{count} pairwise alignments remaining.")
        elif params.matching_params["banality_auto_detection"] is True:
            if params.matching_params["zero_shot_banality_detection"] is True:
                print("Running zero-shot banality filtering...")
                banalities_found = await zero_shot_banality_detection(
                    results_file,
                    params.matching_params["zero_shot_model"],
                    params.matching_params["store_banalities"],
                )
            else:
                banalities_found = banality_auto_detect(
                results_file,
                params.paths["source"]["common_ngrams"],
                f'{params.paths["source"]["ngram_output_path"]}/ngrams_in_order',
                params.matching_params["store_banalities"],
                count,
                params.matching_params["most_common_ngram_proportion"],
                params.matching_params["common_ngram_threshold"],
            )
            if params.matching_params["banality_llm_post_eval"] is True:
                print("Running LLM post-evaluation on flagged banalities...")
                rescued_count = await banality_llm_post_eval(
                    results_file,
                    params.llm_params["server_url"],
                    params.llm_params["server_model"],
                    params.llm_params["context_window"],
                    params.llm_params["concurrency_limit"],
                )
                if rescued_count > 0:
                    print(f"{rescued_count} passages were rescued (reclassified as substantive) after LLM evaluation.")
                    banalities_found -= rescued_count  # Adjust the count
            if params.matching_params["store_banalities"] is False:
                print(
                    f"{banalities_found} pairwise alignment(s) have been identified as formulaic and have been removed from matches."
                )
                count = update_count(count, banalities_found, params.output_path)
                print(f"{count} pairwise alignments remaining.")
            else:
                print(
                    f"{banalities_found} pairwise alignments identified as formulaic and will be flagged as banalities in the database."
                )

    # Passage classification
    if params.passage_classification.get("classify_passage") is True:
        print(f"\n### Classifying passages into thematic categories ###")
        await classify_passages(
            results_file,
            params.matching_params.get("zero_shot_model", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
            params.passage_classification["classes"],
            min_confidence=0.3,
            top_k=3,
            batch_size=32
        )

    # Passage merger
    print("Grouping passages by source...", end="", flush=True)
    groups_file = merge_alignments(results_file, count)

    if params.web_app_config["skip_web_app"] is False:
        create_web_app(
            results_file,
            params.paths["source"]["metadata_path"],
            params.paths["target"]["metadata_path"],
            count,
            params.dbname,
            params.web_app_config["web_application_directory"],
            params.web_app_config["api_server"],
            params.web_app_config["source_url"],
            params.web_app_config["target_url"],
            params.web_app_config["source_philo_db_path"],
            params.web_app_config["target_philo_db_path"],
            params.matching_params["matching_algorithm"],
            params,
            groups_file=groups_file,
            store_banalities=params.matching_params["store_banalities"],
        )



async def run_vsa_similarity(params) -> None:
    """Run vsa similarity"""
    if params.paths["target"]["ngram_output_path"] == "":  # if path not defined make target like source
        params.paths["target"]["ngram_output_path"] = params.paths["source"]["ngram_output_path"]
    if params.text_parsing["parse_source_files"] is True:
        print("\n### Parsing source files ###")
        parse_files(
            params.paths["source"]["input_files"],
            params.text_parsing["source_file_type"],
            params.paths["source"]["input_source_metadata"],
            params.paths["source"]["parse_output"],
            params.text_parsing["source_words_to_keep"],
            params.preprocessing_params["source"]["text_object_type"],
            params.preprocessing_params["source"]["lowercase"],
            params.workers,
            params.debug,
        )
    if params.text_parsing["parse_target_files"] is True:
        print("\n### Parsing target files ###")
        parse_files(
            params.paths["target"]["input_files"],
            params.text_parsing["target_file_type"],
            params.paths["target"]["input_target_metadata"],
            params.paths["target"]["parse_output"],
            params.text_parsing["target_words_to_keep"],
            params.preprocessing_params["target"]["text_object_type"],
            params.preprocessing_params["target"]["lowercase"],
            params.workers,
            params.debug,
        )
    print("\n### Starting vector space alignment ###")
    await run_vsa(
        params.paths["source"]["input_files_for_ngrams"],
        params.paths["target"]["input_files_for_ngrams"],
        params.workers,
        {**params.preprocessing_params, **params.matching_params},
        params.output_path,
        params.llm_params
    )

    # Passage classification (if enabled)
    if params.passage_classification.get("classify_passage") is True:
        output_file = os.path.join(params.output_path, "results/alignments.jsonl.lz4")
        print(f"\n### Classifying passages into thematic categories ###")
        await classify_passages(
            output_file,
            params.matching_params["zero_shot_model"],
            params.passage_classification["classes"],
            min_confidence=0.3,
            top_k=3,
            batch_size=32
        )

    if params.web_app_config["skip_web_app"] is False:
        output_file = os.path.join(params.output_path, "results/alignments.jsonl.lz4")
        count = get_count(os.path.join(params.output_path, "results/counts.txt"))
        create_web_app(
            output_file,
            params.paths["source"]["metadata_path"],
            params.paths["target"]["metadata_path"],
            count,
            params.dbname,
            params.web_app_config["web_application_directory"],
            params.web_app_config["api_server"],
            params.web_app_config["source_url"],
            params.web_app_config["target_url"],
            params.web_app_config["source_philo_db_path"],
            params.web_app_config["target_philo_db_path"],
            params.matching_params["matching_algorithm"],
            params
        )


async def main():
    """Main entry point for the textpair CLI."""
    params = get_config()
    if params.delete is True:
        delete_database(params.dbname)
    elif params.update_db is True:
        count = get_count(os.path.join(params.output_path, "results/count.txt"))
        groups_file = None
        if params.matching_params["matching_algorithm"] == "sa":  # we merge alignments prior to loading
            print("Grouping passages by source...", end="", flush=True)
            groups_file = merge_alignments(params.file, count)
        create_web_app(
            params.file,
            params.paths["source"]["metadata_path"],
            params.paths["target"]["metadata_path"],
            count,
            params.dbname,
            params.web_app_config["web_application_directory"],
            params.web_app_config["api_server"],
            params.web_app_config["source_url"],
            params.web_app_config["target_url"],
            params.web_app_config["source_philo_db_path"],
            params.web_app_config["target_philo_db_path"],
            params.matching_params["matching_algorithm"],
            params,
            load_only_db=True,
            groups_file=groups_file,
            store_banalities=params.matching_params["store_banalities"],
        )
    elif params.only_web_app is True:
        count = get_count(os.path.join(params.output_path, "results/count.txt"))
        groups_file = None
        if params.matching_params["matching_algorithm"] == "sa":  # we merge alignments prior to loading
            print("Grouping passages by source...", end="", flush=True)
            groups_file = merge_alignments(params.file, count)
        create_web_app(
            params.file,
            params.paths["source"]["metadata_path"],
            params.paths["target"]["metadata_path"],
            count,
            params.dbname,
            params.web_app_config["web_application_directory"],
            params.web_app_config["api_server"],
            params.web_app_config["source_url"],
            params.web_app_config["target_url"],
            params.web_app_config["source_philo_db_path"],
            params.web_app_config["target_philo_db_path"],
            params.matching_params["matching_algorithm"],
            params,
            groups_file=groups_file,
            store_banalities=params.matching_params["store_banalities"],
        )
    elif params.matching_params["matching_algorithm"] == "sa":
        await run_alignment(params)
    elif params.matching_params["matching_algorithm"] == "vsa":
        await run_vsa_similarity(params)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
