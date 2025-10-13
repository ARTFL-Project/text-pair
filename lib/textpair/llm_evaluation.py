"""
LLM evaluation module for passage similarity assessment.

This module provides classes and functions for evaluating passage similarity
using Large Language Models (LLMs) via HTTP API endpoints.
"""

import asyncio
import atexit
import json
import re
import subprocess
import time

import aiohttp
import requests
from tqdm import tqdm


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
        source_filename: str,
        target_filename: str,
        source_text: str,
        target_text: str
    ) -> None:
        """Log an LLM evaluation result with comparison to computed similarity."""
        if not self.enabled or not self.llm_file:
            return

        with open(self.llm_file, "a", encoding="utf-8") as debug_file:
            debug_file.write(f"EVALUATION #{index + 1}\n")
            debug_file.write(f"Files: {source_filename} <-> {target_filename}\n")
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

    def __init__(self, model_path: str, context_window: int = 8192, concurrency_limit: int = 8, port: int = 8080):
        self.model_path = model_path
        self.context_window = context_window
        self.port = port
        self.concurrency_limit = concurrency_limit
        self.base_url = f"http://127.0.0.1:{port}"
        self.server_process = None
        self._session = None

        self.banality_eval_params = {
            "temperature": 0.1,
            "max_tokens": 128,
        }

    def start_server(self):
        """Start the llama-server process"""
        # Use the textpair_llama_server command
        cmd = ["textpair_llama_server", self.model_path, str(self.port), str(self.context_window), str(self.concurrency_limit)]
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

    async def evaluate_batch(self, passage_pairs: list[tuple[str, str]], batch_size: int = 8) -> list[tuple[float, str]]:
        """
        Evaluate multiple passage pairs concurrently
        Returns: List of (similarity_score, reasoning) tuples
        """

        async def evaluate_single(session, source_text, target_text):
            try:
                # Create evaluation prompt
                prompt = create_similarity_evaluation_prompt(source_text, target_text, self.context_window)

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

    async def _make_completion_request(
        self,
        prompt: str,
        llm_params: dict
    ) -> dict:
        """
        Helper method to make a completion API request for banality classification.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        payload = {
            "prompt": prompt,
            **llm_params  # All LLM params passed through
        }

        try:
            # Check if server is still running
            if self.server_process and self.server_process.poll() is not None:
                print(f"ERROR: LLM server process has died! Exit code: {self.server_process.poll()}")
                raise Exception("LLM server process is not running")

            async with self._session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60.0)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"DEBUG: Server returned status {response.status}")
                    print(f"DEBUG: Response: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                return await response.json()
        except aiohttp.ServerDisconnectedError as e:
            print(f"DEBUG: Server disconnected during request")
            print(f"DEBUG: Prompt length: {len(prompt)} chars")
            print(f"DEBUG: LLM params: {llm_params}")
            if self.server_process:
                print(f"DEBUG: Server process alive: {self.server_process.poll() is None}")
            raise


    async def score_scholarly_interest(
        self,
        passage: str
    ) -> tuple[int, bool]:
        """
        Score scholarly interest (1-100) for a single passage.

        Args:
            passage: Text passage to evaluate

        Returns:
            Tuple of (score, is_banal) where:
                - score: Interest score (1-100), or -1 on error
                - is_banal: True if score >= 40, False otherwise
        """
        # Create a temporary alignment dict for the prompt
        temp_alignment = {"target_passage": passage}
        prompt = create_scoring_prompt(temp_alignment)

        try:
            result = await self._make_completion_request(prompt, self.banality_eval_params)

            if result.get("choices") and len(result["choices"]) > 0:
                generated_text = result["choices"][0].get("text", "").strip()
            else:
                generated_text = "API_STRUCTURE_ERROR"

            score = -1
            try:
                match = re.search(r"\d+", generated_text)
                if match:
                    score = int(match.group(0))
                    if not (1 <= score <= 100):
                        score = -1
            except Exception:
                pass

            # Score < 40 means banal (not interesting), >= 40 means scholarly/interesting (not banal)
            is_banal = score < 40 if score != -1 else True

            return score, is_banal

        except Exception as e:
            print(f"WARNING: Error scoring passage: {type(e).__name__}: {str(e)}")
            return -1, False

    async def score_scholarly_interest_batch(
        self,
        passages: list[str],
        batch_size: int = 4,
        show_progress: bool = True
    ) -> list[tuple[int, bool]]:
        """
        Batch process passages for scholarly interest scoring.

        Args:
            passages: List of text passages to evaluate
            batch_size: Number of concurrent LLM requests
            show_progress: Show tqdm progress bar

        Returns:
            List of (score, is_banal) tuples in same order as input passages
        """
        async def score_single(session, passage):
            try:
                # Create a temporary alignment dict for the prompt
                temp_alignment = {"target_passage": passage}
                prompt = create_scoring_prompt(temp_alignment)

                payload = {
                    "prompt": prompt,
                    **self.banality_eval_params
                }

                async with session.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60.0)
                ) as response:
                    if response.status != 200:
                        return -1, False

                    result = await response.json()

                    if result.get("choices") and len(result["choices"]) > 0:
                        generated_text = result["choices"][0].get("text", "").strip()
                    else:
                        return -1, False

                    score = -1
                    try:
                        match = re.search(r"\d+", generated_text)
                        if match:
                            score = int(match.group(0))
                            if not (1 <= score <= 100):
                                score = -1
                    except Exception:
                        pass

                    # Score < 40 means banal (not interesting), >= 40 means scholarly/interesting (not banal)
                    is_banal = score < 40 if score != -1 else True

                    return score, is_banal

            except Exception as e:
                return -1, False

        results = []
        total_passages = len(passages)

        with tqdm(total=total_passages, desc="LLM Scholarly Evaluation", unit="passages", disable=not show_progress) as pbar:
            for i in range(0, total_passages, batch_size):
                batch = passages[i:i + batch_size]

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, passage) for passage in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in batch_results:
                        if isinstance(result, Exception):
                            results.append((-1, False))
                        else:
                            results.append(result)
                        pbar.update(1)

        return results



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


# ============================================================================
# Prompt Creation Functions
# ============================================================================


def create_scoring_prompt(alignment: dict) -> str:
    """Create prompt for Stage 2 scholarly interest scoring"""
    prompt = f"""Evaluate the likely scholarly interest of the following text passage, which has been identified as a text reuse. The goal is to distinguish reuses that represent substantive content or unique expression from those that are merely conventional, boilerplate, structural, or formulaic elements often found repeated in texts.
    Assign a score from 1 to 100 based on these criteria:

    LOW SCORE (e.g., 1-40 => Likely Not Scholarly Interesting as Reuse):

    Standard greetings, closings, simple sign-offs.
    Standard correspondence headers (e.g., 'Sender, Location, to Recipient, Location', date lines).
    Bibliographic citations (author, title, etc.) or common publishing notes (e.g., 'Printed by...', 'Composé et mis...').
    Simple lists of names, places, items with minimal context. Historical significance is not enough. Lists of kings, emperors, historical or religious figures are not interesting.
    The simple description of a historical function (e.g., founder of, inventor of, emperor of) should be considered a common way of describing a historical figure, and is therefore repetitive and not particularly interesting.
    Highly repetitive legal or administrative boilerplate/formalities (e.g., standard parts of decrees, registration clauses like 'Royale des Libraires...'). Remark that lists of dignitaries and titles (following the model: nos amez & feaux Conseillers, les gens tenans nos Cours de Parlement, etc.) belong to this category, in fact, they are nothing more than the recipients of royal decrees. All these types of reuse should be considered with the lowest possible score.
    Simple titles of persons when listed alone, or their mutual family relationships are not interesting at all.
    Any rhetorical nuance (adjectives, adverbs) added to a standard and banal structure does not make reuse any more interesting (e.g. très humble)
    If it is an opening or closing formula, remember that expressions of wishes and hopes are also highly rhetorical and formulaic, and should therefore be evaluated with a very low score.
    A mere signature does not contribute any meaningful relevance to the excerpt

    MID-RANGE SCORE (e.g., 40-60):

    Common proverbs, aphorisms, or less common formulas.
    Borderline cases or passages with minor substantive content mixed with formulaic elements.
    Very common, short religious formulas/prayers used repetitively (e.g., 'Amen', 'Ora pro nobis').

    HIGH SCORE (e.g., 60-100 => Potentially Scholarly Interesting as Reuse):

    Reused passages containing substantive arguments, detailed analysis, or unique opinions/reflections.
    Detailed descriptions, narrative segments, or complex explanations.
    Distinct or particularly well-phrased expressions of ideas.
    Longer, significant quotations (from literature, philosophy, scripture, historical figures) likely reused for their specific content or meaning.
    The core text of specific laws, decrees, or oaths where the substantive content is the focus.
    Passages, regardless of length or formulaic nature, that represent a pivotal, explicit statement of a significant philosophical, theological, or socio-political concept or axiom. Reuse of such passages indicates a core intellectual lineage, widespread transmission of a major idea, or engagement with fundamental historical thought.

    Consider the passage's complexity, length, and apparent function within a text. Short, highly predictable, purely functional text should score low. Evaluate the provided passage on its own merits based on these criteria.


    INSTRUCTIONS:
    1. Provide a score (integer 1-100). Carefully consider all criteria above when assigning the score.
    2. Provide a brief (one sentence) explanation justifying the score based ONLY on the criteria above (substantive vs. conventional/formulaic/structural).

    Answer ONLY in the following specific format:
    [SCORE]: [Brief explanation]

    Example Output 1:
    85: Substantive philosophical argument about free will using unique phrasing.
    Example Output 2:
    10: Standard correspondence closing formula with no unique content.
    Example Output 3:
    45: Common proverb reused without further analysis.

    PASSAGE: "{alignment['target_passage']}"
    """
    return prompt


def create_similarity_evaluation_prompt(source_text: str, target_text: str, context_window: int) -> str:
    """Create prompt for the LLM to evaluate passage similarity"""
    # Truncate texts if too long based on self.context_window
    # we approximate number of tokens as number of characters / 4
    max_chars = context_window * 4 // 3  # Leave room for prompt
    if len(source_text) + len(target_text) > max_chars:
        half_max = max_chars // 2
        source_text = source_text[:half_max] + "..."
        target_text = target_text[:half_max] + "..."

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
