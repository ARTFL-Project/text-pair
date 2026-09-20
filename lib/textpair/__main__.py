#!/usr/bin/env python3
"""Sequence aligner script"""

import configparser
import os
import shutil
import subprocess
import sys
from shlex import quote

import psycopg2

from . import get_config
from .parse_config import read_global_config
from .sequence_alignment import (
    Ngrams,
    banality_auto_detect,
    banality_llm_post_eval,
    filter_and_flag,
    merge_alignments,
    phrase_matcher,
    separate_banalities,
)


def build_graph_and_labels(
    alignments_file: str,
    embedding_model: str,
    graph_params: dict | None = None,
    preprocessing_params: dict | None = None,
    sweep: bool = False,
) -> None:
    """
    Build the thematic identity graph and label its clusters.

    Runs in the separate graph environment via subprocess, because UMAP/HDBSCAN
    (and optionally the RAPIDS stack) conflict with the main pipeline's deps.

    Labeling is c-TF-IDF term extraction followed by `topologic-labeler`; it
    degrades to top-word descriptions when that tool is not installed, so no
    LLM server configuration is required here.
    """
    graph_params = graph_params or {}
    preprocessing_params = preprocessing_params or {}

    if not embedding_model:
        print(
            "ERROR: [GRAPH] build_graph is enabled but no embedding model is set.\n"
            "       Set embedding_model under [GRAPH], or under [PREPROCESSING] for a\n"
            "       vector-space run to share it with alignment. Skipping the graph.",
            file=sys.stderr,
        )
        return

    print("\n### Building thematic identity graph ###", flush=True)

    graph_python = "/var/lib/text-pair/graph/bin/python"
    output_dir = os.path.dirname(alignments_file)

    build_command = [
        graph_python,
        "-m",
        "textpair_graph",
        "build",
        alignments_file,
        output_dir,
        "--model",
        embedding_model,
    ]
    mcs = graph_params.get("min_cluster_size", "auto")
    build_command += ["--min-cluster-size", str(mcs)]
    build_command += ["--cluster-selection-method", graph_params.get("cluster_selection_method", "eom")]
    if not graph_params.get("merge_unthemed", True):
        build_command.append("--no-merge-unthemed")
    for side in ("source", "target"):
        field = graph_params.get(f"{side}_author_field")
        if field:
            build_command += [f"--{side}-author-field", field]
    if sweep:
        build_command.append("--sweep")

    try:
        # No success message here: the subprocess writes straight to this
        # terminal and already reports it, with the output path.
        subprocess.run(build_command, check=True, capture_output=False)
        if sweep:
            # --sweep prints the candidate table and writes nothing, so there is
            # no clustering to label.
            return
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: Graph model generation failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        return
    except FileNotFoundError:
        print(f"ERROR: Graph environment not found at {graph_python}", file=sys.stderr)
        print("Install textpair_graph in a separate environment to enable graph functionality.")
        return

    print("\n### Labeling clusters (c-TF-IDF + LLM) ###", flush=True)
    graph_data_path = os.path.join(output_dir, "graph_data")
    if not os.path.exists(graph_data_path):
        print(f"WARNING: {graph_data_path} not found; skipping labeling.", file=sys.stderr)
        return

    label_command = [
        graph_python,
        "-m",
        "textpair_graph",
        "label",
        graph_data_path,
        "--alignments-file",
        alignments_file,
        "--model",
        graph_params.get("label_model", "google/gemma-4-E2B-it"),
        "--language",
        graph_params.get("label_language", "French"),
    ]
    if preprocessing_params.get("language"):
        label_command += ["--text-language", preprocessing_params["language"]]
    # The graph's own spacy_model is the lemmatizer for term extraction, which is
    # a different model from the alignment pipeline's language_model.
    spacy_model = graph_params.get("spacy_model") or preprocessing_params.get("language_model")
    if spacy_model:
        label_command += ["--spacy-model", spacy_model]
    if preprocessing_params.get("pos_to_keep"):
        label_command += ["--pos-to-keep", ",".join(preprocessing_params["pos_to_keep"])]
    # Term extraction modernizes by default. Forward the alignment's setting so a
    # run that deliberately keeps period spelling is not modernized behind its back.
    if not preprocessing_params.get("modernize", True):
        label_command.append("--no-modernize")

    try:
        subprocess.run(label_command, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(
            f"WARNING: Cluster labeling failed with exit code {e.returncode}",
            file=sys.stderr,
        )


def delete_database(dbname: str) -> None:
    global_config = read_global_config()
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
        shutil.rmtree(os.path.join(global_config["WEB_APP"]["web_app_path"], dbname), ignore_errors=True)
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


def run_python_aligner(params) -> None:
    """Run the Python sequence aligner in-process with the same parameters as the binary."""
    from .sequence_alignment.aligner import align

    print("Using the Python aligner.", flush=True)
    align(
        f'{params.paths["source"]["ngram_output_path"]}/ngrams',
        params.paths["source"]["metadata_path"],
        f"{params.output_path}/results",
        target_files=f'{params.paths["target"]["ngram_output_path"]}/ngrams',
        target_metadata=params.paths["target"]["metadata_path"],
        threads=params.workers,
        sort_by=params.matching_params["sort_by"],
        source_batch=params.matching_params["source_batch"],
        target_batch=params.matching_params["target_batch"],
        matching_window_size=params.matching_params["matching_window_size"],
        max_gap=params.matching_params["max_gap"],
        flex_gap=params.matching_params["flex_gap"],
        minimum_matching_ngrams=params.matching_params["minimum_matching_ngrams"],
        minimum_matching_ngrams_in_window=params.matching_params["minimum_matching_ngrams_in_window"],
        minimum_matching_ngrams_in_docs=params.matching_params["minimum_matching_ngrams_in_docs"],
        context_size=params.matching_params["context_size"],
        duplicate_threshold=params.matching_params["duplicate_threshold"],
        merge_passages_on_byte_distance=params.matching_params["merge_passages_on_byte_distance"],
        merge_passages_on_ngram_distance=params.matching_params["merge_passages_on_ngram_distance"],
        passage_distance_multiplier=params.matching_params["passage_distance_multiplier"],
        debug=params.debug,
        ngram_index=params.matching_params["ngram_index"],
    )


async def run_alignment(params):
    """Main function to start sequence alignment"""
    from . import classify_passages, create_web_app, parse_files

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
        shutil.rmtree(result_batch_path, ignore_errors=True)
    results_file = f"{params.output_path}/results/alignments.jsonl.lz4"
    if os.path.exists(results_file):
        os.remove(results_file)
    run_python_aligner(params)
    if len(os.listdir(result_batch_path)) == 1:
        filename = os.listdir(result_batch_path)[0]
        shutil.move(os.path.join(result_batch_path, filename), results_file)
        shutil.rmtree(result_batch_path, ignore_errors=True)
    else:
        print(
            "Merging alignments into one file (this may take a while)... ",
            end="",
            flush=True,
        )
        # NUL-delimited so batch paths containing spaces survive the pipeline.
        merge_command = (
            f"find {quote(result_batch_path)} -type f -print0 | sort -zV | "
            f"xargs -0 lz4cat --rm | lz4 -q > {quote(results_file)}"
        )
        os.system(merge_command)
        shutil.rmtree(result_batch_path, ignore_errors=True)
        print("done.")
    count = get_count(os.path.join(params.output_path, "results/count.txt"))

    # Postprocessing steps
    if any(
        [
            params.matching_params["phrase_filter"],
            params.matching_params["banality_auto_detection"],
            params.matching_params["banality_llm_eval"],
        ]
    ):
        print(f"\n### Postprocessing {count} pairwise alignments ###")
        # The phrase list removes known boilerplate and auto-detection catches
        # the rest, so they are independent verdicts -- but on the same records,
        # and run one after the other they read and rewrite the whole result
        # file twice. With both on they share a pass.
        phrase_filter = params.matching_params["phrase_filter"]
        auto_detect = params.matching_params["banality_auto_detection"] is True
        ngrams_in_order = f"{params.paths['source']['ngram_output_path']}/ngrams_in_order"
        filtered_passages = 0
        banalities_found = 0
        if phrase_filter and auto_detect:
            print("Running phrase filter and automatic banality detection...")
            filtered_passages, banalities_found = filter_and_flag(
                results_file,
                phrase_filter,
                params.paths["source"]["common_ngrams"],
                ngrams_in_order,
                count,
                params.matching_params["most_common_ngram_proportion"],
                params.matching_params["common_ngram_threshold"],
            )
        elif phrase_filter:
            filtered_passages = phrase_matcher(results_file, phrase_filter, count)
        elif auto_detect:
            print("Running automatic banality detection...")
            banalities_found = banality_auto_detect(
                results_file,
                params.paths["source"]["common_ngrams"],
                ngrams_in_order,
                params.matching_params["store_banalities"],
                count,
                params.matching_params["most_common_ngram_proportion"],
                params.matching_params["common_ngram_threshold"],
            )
        if phrase_filter:
            print(f"{filtered_passages} pairwise alignments have been filtered based on the phrase filter provided.")
            count = update_count(count, filtered_passages, params.output_path)
            print(f"{count} pairwise alignments remaining.")
        if auto_detect:
            print(f"{banalities_found} pairwise alignment(s) have been identified as formulaic.")
            if params.matching_params["banality_llm_post_eval"] is True:
                print("Running LLM post-evaluation on flagged banalities...")
                rescued_count = await banality_llm_post_eval(
                    results_file,
                    params.llm_params.get("llm_model", ""),
                    params.llm_params["llm_context_window"],
                    params.llm_params["llm_concurrency_limit"],
                    params.llm_params.get("llm_port", 8080),
                    params.matching_params["store_banalities"],
                    base_url=params.llm_params.get("llm_base_url", ""),
                    api_key=params.llm_params.get("llm_api_key", ""),
                )
                if rescued_count > 0:
                    print(f"{rescued_count} passages were rescued (reclassified as substantive) after LLM evaluation.")
                    banalities_found -= rescued_count  # Adjust the count
            if params.matching_params["store_banalities"] is False:
                # Separate banalities into a different file after all evaluation is complete
                banalities_found = separate_banalities(results_file, count)
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
    if params.passage_classification["classify_passage"] is True:
        print(f"\n### Classifying passages into thematic categories ###")
        await classify_passages(
            results_file,
            params.passage_classification["zero_shot_model"],
            params.passage_classification["classes"],
            min_confidence=0.3,
            top_k=3,
            batch_size=32,
        )

    # Passage merger
    print("Grouping passages by source...", end="", flush=True)
    groups_file = merge_alignments(results_file, count)

    if params.web_app_config["skip_web_app"] is False:
        if params.graph_params.get("build_graph"):
            build_graph_and_labels(
                results_file,
                params.graph_params.get("embedding_model", ""),
                graph_params=params.graph_params,
                preprocessing_params=params.preprocessing_params["source"],
            )

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
    from . import classify_passages, create_web_app, parse_files, run_vsa

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
        params.llm_params,
    )

    # Passage classification (if enabled)
    if params.passage_classification["classify_passage"] is True:
        output_file = os.path.join(params.output_path, "results/alignments.jsonl.lz4")
        print(f"\n### Classifying passages into thematic categories ###")
        await classify_passages(
            output_file,
            params.passage_classification["zero_shot_model"],
            params.passage_classification["classes"],
            min_confidence=0.5,
            top_k=3,
            batch_size=32,
        )

    if params.web_app_config["skip_web_app"] is False:
        output_file = os.path.join(params.output_path, "results/alignments.jsonl.lz4")
        count = get_count(os.path.join(params.output_path, "results/counts.txt"))

        if params.graph_params.get("build_graph"):
            build_graph_and_labels(
                output_file,
                params.graph_params.get("embedding_model", ""),
                graph_params=params.graph_params,
                preprocessing_params=params.preprocessing_params["source"],
            )

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
            params,
        )


async def main():
    """Main entry point for the textpair CLI."""
    from . import create_web_app

    params = get_config()

    # Save a copy of the config file to the output directory for reproducibility
    config_file = params.config
    if config_file and os.path.exists(config_file):
        os.makedirs(params.output_path, exist_ok=True)
        shutil.copy2(config_file, os.path.join(params.output_path, f"{params.dbname}_config.ini"))

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
    elif params.graph_only is True:
        # Rebuild the graph from an existing alignment and publish it, using the
        # same [GRAPH] settings as a full run so the two cannot drift apart.
        alignments_file = params.file or os.path.join(params.output_path, "results", "alignments.jsonl.lz4")
        if not os.path.exists(alignments_file):
            print(f"ERROR: no alignment file at {alignments_file}", file=sys.stderr)
            print("Pass --file to point at one.", file=sys.stderr)
            sys.exit(1)
        build_graph_and_labels(
            alignments_file,
            params.graph_params.get("embedding_model", ""),
            graph_params=params.graph_params,
            preprocessing_params=params.preprocessing_params.get("source", {}),
            sweep=params.sweep,
        )
        if params.sweep is False and params.skip_web_app is False:
            from .web_loader import publish_graph_data

            db_dir = os.path.join(params.web_app_config["web_application_directory"], params.dbname)
            if os.path.isdir(db_dir):
                publish_graph_data(os.path.join(os.path.dirname(alignments_file), "graph_data"), db_dir)
            else:
                print(f"Note: no web app at {db_dir}; graph built but not published.")
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


def run():
    """Sync entry point for console_scripts."""
    import asyncio
    import platform

    # macOS defaults to 256 open file descriptors, which is too low for
    # PhiloLogic's sort/merge operations on large corpora; raise it for
    # the duration of the run and restore it afterwards.
    original_soft = None
    hard = None
    if platform.system() == "Darwin":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 4096:
            new_soft = min(hard, 10240)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            original_soft = soft
            print(f"[macOS] Raised open file limit: {soft} -> {new_soft}")
    try:
        asyncio.run(main())
    finally:
        if original_soft is not None:
            import resource

            resource.setrlimit(resource.RLIMIT_NOFILE, (original_soft, hard))
            print(f"[macOS] Restored open file limit: {original_soft}")


if __name__ == "__main__":
    run()
