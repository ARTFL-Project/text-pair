"""
LLM evaluation module for passage similarity assessment.

This module provides classes and functions for evaluating passage similarity
using Large Language Models (LLMs) via HTTP API endpoints.
"""

import asyncio
import atexit
import re
import subprocess
import time

import aiohttp
import requests
from tqdm import tqdm

from .structures import MergedGroup


class LLMDebugLogger:
    """
    Debug logger for LLM operations, providing structured logging for LLM evaluations
    and comparison with computed similarities.
    """
    def __init__(self, enabled: bool = False, output_path: str = "output"):
        self.enabled = enabled
        self.output_path = output_path

        if self.enabled:
            import os
            os.makedirs(self.output_path, exist_ok=True)
            self.llm_file = f"{self.output_path}/llm_debug.log"
        else:
            self.llm_file = None

    def log_llm_evaluation(
        self,
        index: int,
        computed_similarity: float,
        llm_similarity: float,
        llm_reasoning: str,
        merged_group: MergedGroup,
        source_text: str,
        target_text: str
    ) -> None:
        """Log an LLM evaluation result with comparison to computed similarity."""
        if not self.enabled or not self.llm_file:
            return

        with open(self.llm_file, "a", encoding="utf-8") as debug_file:
            debug_file.write(f"EVALUATION #{index + 1}\n")
            debug_file.write(f"Files: {merged_group.source.filename} <-> {merged_group.target.filename}\n")
            debug_file.write(f"Computed Similarity: {computed_similarity:.3f}\n")
            debug_file.write(f"LLM Similarity: {llm_similarity:.3f}\n")
            debug_file.write(f"LLM Reasoning: {llm_reasoning}\n")
            debug_file.write("Source Text:\n")
            debug_file.write(f"{source_text[:200]}{'...' if len(source_text) > 200 else ''}\n")
            debug_file.write("Target Text:\n")
            debug_file.write(f"{target_text[:200]}{'...' if len(target_text) > 200 else ''}\n")
            debug_file.write("-" * 80 + "\n\n")

    def log_llm_error(self, error_message: str) -> None:
        """Log an LLM evaluation error."""
        if not self.enabled or not self.llm_file:
            return

        with open(self.llm_file, "a", encoding="utf-8") as debug_file:
            debug_file.write(f"LLM ERROR: {error_message}\n")
            debug_file.write("-" * 80 + "\n\n")

    def log_expansion_debug(
        self,
        round_num: int,
        start_expansion: int,
        end_expansion: int,
        similarity: float,
        reasoning: str,
        expansion_type: str
    ) -> None:
        """Log expansion evaluation details."""
        if not self.enabled or not self.llm_file:
            return

        with open(self.llm_file, "a", encoding="utf-8") as debug_file:
            debug_file.write(f"EXPANSION ROUND {round_num} ({expansion_type})\n")
            debug_file.write(f"Expansion: {start_expansion} -> {end_expansion}\n")
            debug_file.write(f"Similarity: {similarity:.3f}\n")
            debug_file.write(f"Reasoning: {reasoning}\n")
            debug_file.write("-" * 40 + "\n\n")


class AsyncLLMEvaluator:
    """Async LLM-based similarity evaluator using llama-server via HTTP"""

    def __init__(self, model_path: str, context_window: int = 8192, port: int = 8080):
        self.model_path = model_path
        self.context_window = context_window
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.server_process = None

    def start_server(self):
        """Start the llama-server process"""
        # Use the textpair_llama_server command
        cmd = ["textpair_llama_server", self.model_path, str(self.port), str(self.context_window)]
        self.server_process = subprocess.Popen(cmd)

        # Wait for server to be ready
        self._wait_for_server()
        atexit.register(self.stop_server)

    def _wait_for_server(self):
        """Wait for server to be ready"""
        max_retries = 30
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        raise RuntimeError("Failed to start llama-server")

    def stop_server(self):
        """Stop the llama-server process"""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None

    async def evaluate_similarity(self, source_text: str, target_text: str) -> tuple[float, str]:
        """
        Evaluate passage similarity using LLM via HTTP
        Returns: (similarity_score, reasoning)
        """
        try:

            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(source_text, target_text)

            # Prepare request payload
            payload = {
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.1,
                "stop": []
            }

            # Make async HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")

                    result = await response.json()

            # Extract response text
            response_text = ""
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                response_text = choice.get('text', '')

            response_text = str(response_text).strip()
            score, reasoning = self._parse_llm_response(response_text)

            return score, reasoning

        except Exception as e:
            error_msg = f"Async LLM evaluation failed: {str(e)[:100]}..."
            return 0.0, f"Error: {error_msg}"

    async def evaluate_batch(self, passage_pairs: list[tuple[str, str]], batch_size: int = 8) -> list[tuple[float, str]]:
        """
        Evaluate multiple passage pairs concurrently
        Returns: List of (similarity_score, reasoning) tuples
        """
        import aiohttp

        async def evaluate_single(session, source_text, target_text):
            try:
                # Create evaluation prompt
                prompt = self._create_evaluation_prompt(source_text, target_text)

                # Prepare request payload
                payload = {
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "stop": []
                }

                # Make async HTTP request
                async with session.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")

                    result = await response.json()

                # Extract response text
                response_text = ""
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    response_text = choice.get('text', '')

                response_text = str(response_text).strip()
                return self._parse_llm_response(response_text)

            except Exception as e:
                return 0.0, f"Error: {str(e)[:100]}..."

        # Process in batches to avoid overwhelming the server
        results = []
        total_pairs = len(passage_pairs)

        with tqdm(total=total_pairs, desc="LLM Evaluation", unit="pairs", leave=False) as pbar:
            for i in range(0, len(passage_pairs), batch_size):
                batch = passage_pairs[i:i + batch_size]

                async with aiohttp.ClientSession() as session:
                    tasks = [
                        evaluate_single(session, source, target)
                        for source, target in batch
                    ]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Handle any exceptions and update progress
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            results.append((0.0, f"Error: {str(result)[:100]}..."))
                        else:
                            results.append(result)

                        # Update progress bar
                        pbar.update(1)

        return results

    def _create_evaluation_prompt(self, source_text: str, target_text: str) -> str:
        """Create evaluation prompt for the LLM"""
        # Truncate texts if too long
        max_text_length = 400  # Shorter to ensure model can respond
        if len(source_text) > max_text_length:
            source_text = source_text[:max_text_length] + "..."
        if len(target_text) > max_text_length:
            target_text = target_text[:max_text_length] + "..."

        prompt = f"""You are a text analysis expert. Your task is to rate the semantic similarity of two passages.

First, determine if the passages address the same specific argument. Then, use the score guide below.

IMPORTANT: Direct agreement and direct disagreement on the exact same point are both forms of HIGH similarity.
IMPORTANT: Avoid defaulting to the boundary scores of a category (like 0.40, 0.70, or 0.90). Use the full range to show nuance.

Score Guide:
• 0.0 - 0.4: Different Subjects. The passages are about completely different topics.
• > 0.4 to < 0.7: Shared Subject, Different Focus. The passages are about the same broad subject (e.g., the Roman Empire) but focus on different specific arguments or aspects (e.g., one is about military tactics, the other about trade policy).
• 0.7 - 0.9: Shared Subject, Shared Focus. The passages address the exact same specific argument, question, or thesis. They are in direct conversation, whether they agree, disagree, or analyze it in parallel.
• > 0.9 - 1.0: Paraphrase. The passages make the exact same point and have nearly identical meaning.

Your thought process:
1. What is the broad subject of each passage?
2. Do they narrow in on the exact same specific argument or point?
3. Based on that, which score category do they fall into?

Provide your answer in this exact format:
Score: X.XX
Reasoning: [Your step-by-step analysis]

---
Passage 1: {source_text}
---
Passage 2: {target_text}
---
Answer:"""

        return prompt

    def _parse_llm_response(self, response: str) -> tuple[float, str]:
        """Parse LLM response to extract score and reasoning"""
        try:
            # Try multiple score patterns
            score_patterns = [
                r'Score:\s*([0-9]*\.?[0-9]+)',  # "Score: 0.8"
                r'score:\s*([0-9]*\.?[0-9]+)',  # "score: 0.8" (lowercase)
                r'([0-9]*\.?[0-9]+)',  # Just a number anywhere
            ]

            score = 0.0
            for pattern in score_patterns:
                score_match = re.search(pattern, response, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))  # Clamp to valid range
                    break

            # Try multiple reasoning patterns
            reasoning_patterns = [
                r'Reasoning:\s*(.+)',  # "Reasoning: explanation"
                r'reasoning:\s*(.+)',  # "reasoning: explanation" (lowercase)
                r'because\s*(.+)',     # "because explanation"
                r'since\s*(.+)',       # "since explanation"
            ]

            reasoning = "No reasoning provided"
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    reasoning = reasoning
                    break

            # If no specific reasoning found, use the whole response as reasoning
            if reasoning == "No reasoning provided" and response.strip():
                reasoning = response.strip()

            return score, reasoning

        except Exception as e:
            return 0.0, f"Failed to parse LLM response: {str(e)}"


async def evaluate_passages_with_llm(
    merged_matches: list[MergedGroup],
    min_score: float,
    llm_model_path: str,
    llm_context_window: int,
    llm_similarity_threshold: float | None = None,
    debug_llm: bool = False,
    output_path: str = "output",
    get_text_func=None,
) -> tuple[list[MergedGroup], AsyncLLMEvaluator]:
    """Evaluate merged passages using LLM and filter by threshold. Returns (matches, evaluator)."""

    # Initialize LLM evaluator
    llm_evaluator = AsyncLLMEvaluator(llm_model_path, context_window=llm_context_window)
    llm_evaluator.start_server()

    # Initialize debug logger (controlled by debug_llm parameter)
    debug_logger = LLMDebugLogger(enabled=debug_llm, output_path=output_path)

    print("Evaluating matched passages with LLM...", flush=True)

    # Initialize LLM evaluation debug file if debugging is enabled
    if debug_logger.enabled:
        with open(debug_logger.llm_file, "w", encoding="utf-8") as debug_file:
            debug_file.write("LLM EVALUATION DEBUG LOG\n")
            debug_file.write("=" * 50 + "\n\n")

    # Override min_score if llm_similarity_threshold is provided
    if llm_similarity_threshold is not None:
        min_score = llm_similarity_threshold

    try:
        # Prepare passage pairs for batch evaluation
        passage_pairs = []
        computed_similarities = []

        for merged_group in merged_matches:
            source_text = get_text_func(
                merged_group.source.start_byte,
                merged_group.source.end_byte,
                merged_group.source.filename
            )
            target_text = get_text_func(
                merged_group.target.start_byte,
                merged_group.target.end_byte,
                merged_group.target.filename
            )
            passage_pairs.append((source_text, target_text))
            computed_similarities.append(merged_group.similarity)

        # Perform batch async evaluation
        llm_results = await llm_evaluator.evaluate_batch(passage_pairs, batch_size=8)

        # Update similarities and filter by threshold
        for i, (merged_group, (llm_similarity, llm_reasoning)) in enumerate(zip(merged_matches, llm_results)):
            computed_similarity = computed_similarities[i]

            # Update the similarity score with LLM evaluation
            merged_group.similarity = llm_similarity

            # Debug: Log the LLM evaluation
            debug_logger.log_llm_evaluation(
                i, computed_similarity, llm_similarity, llm_reasoning,
                merged_group, passage_pairs[i][0], passage_pairs[i][1]
            )

        # Filter out passages below threshold
        original_count = len(merged_matches)
        filtered_matches = [match for match in merged_matches if match.similarity >= min_score]

        print(f"Completed LLM evaluation. Kept {len(filtered_matches)}/{original_count} passages above {min_score} threshold.", flush=True)

        return filtered_matches, llm_evaluator

    except Exception as llm_error:
        print(f"Batch LLM evaluation failed: {str(llm_error)[:100]}...")
        # Keep the original similarities as fallback
        debug_logger.log_llm_error(str(llm_error))
        return merged_matches, llm_evaluator