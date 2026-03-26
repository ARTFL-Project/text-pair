"""
LLM evaluation module for passage similarity assessment.

This module provides classes and functions for evaluating passage similarity
using Large Language Models (LLMs) via HTTP API endpoints.
"""

import asyncio
import atexit
import json
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
        target_text: str,
        kept: bool = True,
    ) -> None:
        """Log an LLM evaluation result with comparison to computed similarity."""
        if not self.enabled or not self.llm_file:
            return

        status = "KEPT" if kept else "REJECTED"
        with open(self.llm_file, "a", encoding="utf-8") as debug_file:
            debug_file.write(f"EVALUATION #{index + 1} [{status}]\n")
            debug_file.write(f"Files: {source_filename} <-> {target_filename}\n")
            debug_file.write(f"Computed Similarity: {computed_similarity:.3f}\n")
            debug_file.write(f"LLM Similarity: {llm_similarity:.3f}\n")
            debug_file.write(f"LLM Reasoning: {llm_reasoning}\n")
            debug_file.write("Source Text:\n")
            debug_file.write(f"{source_text}\n")
            debug_file.write("Target Text:\n")
            debug_file.write(f"{target_text}\n")
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
        expansion_type: str,
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

    # JSON schemas for structured output via llama.cpp grammar-constrained generation
    SIMILARITY_SCHEMA = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        },
        "required": ["reasoning", "score"],
    }

    BANALITY_SCHEMA = {
        "type": "object",
        "properties": {
            "score": {"type": "integer"},
            "explanation": {"type": "string"},
        },
        "required": ["score", "explanation"],
    }

    def __init__(
        self,
        model_path: str = "",
        context_window: int = 8192,
        concurrency_limit: int = 8,
        port: int = 8080,
        base_url: str = "",
        api_key: str = "",
    ):
        self.model_path = model_path
        self.context_window = context_window
        self.port = port
        self.concurrency_limit = concurrency_limit
        self.server_process = None
        self._session = None
        self.api_key = api_key

        # If base_url is provided, use external server; otherwise build from port
        self._external = bool(base_url)
        self.base_url = base_url.rstrip("/") if base_url else f"http://127.0.0.1:{port}"

    def start_server(self):
        """Start the llama-server process. Skipped if using an external server."""
        if self._external:
            print(f"Using external LLM server at {self.base_url}")
            self._wait_for_server()
            return

        # Use the textpair_llama_server command
        cmd = [
            "textpair_llama_server",
            self.model_path,
            str(self.port),
            str(self.context_window),
            str(self.concurrency_limit),
        ]
        self.server_process = subprocess.Popen(cmd)

        # Wait for server to be ready
        self._wait_for_server()
        atexit.register(self.stop_server)

    def _wait_for_server(self):
        """Wait for server to be ready"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Try /health (llama.cpp, vLLM) then /v1/models (any OpenAI-compatible API)
        health_endpoints = [f"{self.base_url}/health", f"{self.base_url}/v1/models"]
        max_retries = 5 if self._external else 30
        for attempt in range(max_retries):
            for endpoint in health_endpoints:
                try:
                    response = requests.get(endpoint, headers=headers, timeout=5)
                    if response.status_code == 200:
                        return
                    # Accept 401/403/404 as signs the server is reachable
                    if self._external and response.status_code in (401, 403, 404):
                        return
                except requests.exceptions.RequestException:
                    pass
            time.sleep(1)

        raise RuntimeError(f"LLM server at {self.base_url} failed to become ready")

    def stop_server(self):
        """Stop the llama-server process"""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None

    def _build_payload(self, prompt: str, json_schema: dict, max_tokens: int = 5000) -> dict:
        """Build a chat completions payload with structured JSON output."""
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            },
        }
        # Only include model when set (required for multi-model routers like OpenRouter,
        # unnecessary for single-model servers like llama-server or dedicated vLLM)
        if self.model_path:
            payload["model"] = self.model_path
        return payload

    def _get_headers(self) -> dict:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_text(result: dict) -> str:
        """Extract generated text from a chat completions response."""
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "").strip()
            # Fallback for raw completions format
            return choice.get("text", "").strip()
        return ""

    async def evaluate_batch(
        self, passage_pairs: list[tuple[str, str]], batch_size: int | None = None, show_progress: bool = True
    ) -> list[tuple[float, str, str]]:
        """
        Evaluate multiple passage pairs concurrently
        Returns: List of (similarity_score, reasoning, stance) tuples
        """

        async def evaluate_single(session, source_text, target_text, retry: bool = True):
            try:
                prompt = create_similarity_evaluation_prompt(source_text, target_text, self.context_window)
                payload = self._build_payload(prompt, self.SIMILARITY_SCHEMA)

                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text[:300]}")
                    result = await response.json()

                score, reasoning, stance = self._parse_llm_response(self._extract_text(result))
                if score == 0.0 and retry:
                    return await evaluate_single(session, source_text, target_text, retry=False)
                return score, reasoning, stance

            except asyncio.TimeoutError:
                if retry:
                    return await evaluate_single(session, source_text, target_text, retry=False)
                print(
                    f"WARNING: LLM request timed out after 120s. llm_concurrency_limit is set to {self.concurrency_limit},"
                    f" meaning {self.concurrency_limit} requests are sent to the server simultaneously."
                    f" Either lower llm_concurrency_limit to match the number of slots your server was started with,"
                    f" or increase your server's parallel slot count to {self.concurrency_limit}.",
                    flush=True,
                )
                return 0.0, "Error: timeout", "Unknown"
            except Exception as e:
                if retry:
                    return await evaluate_single(session, source_text, target_text, retry=False)
                print(f"WARNING: LLM evaluation error: {e}", flush=True)
                return 0.0, f"Error: {str(e)[:100]}...", "Unknown"

        # Process in batches to avoid overwhelming the server
        if batch_size is None:
            batch_size = self.concurrency_limit
        results = []
        total_pairs = len(passage_pairs)

        with tqdm(total=total_pairs, desc="LLM Evaluation", unit="pairs", leave=False, disable=not show_progress) as pbar:
            for i in range(0, len(passage_pairs), batch_size):
                batch = passage_pairs[i : i + batch_size]

                async with aiohttp.ClientSession() as session:
                    tasks = [evaluate_single(session, source, target) for source, target in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Handle any exceptions
                    for result in batch_results:
                        if isinstance(result, Exception):
                            results.append((0.0, f"Error: {str(result)[:100]}...", "Unknown"))
                        else:
                            results.append(result)

                    pbar.update(len(batch_results))

        return results

    async def _make_chat_request(self, prompt: str, json_schema: dict, max_tokens: int = 128) -> dict:
        """
        Make a chat completions request with structured JSON output.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        payload = self._build_payload(prompt, json_schema, max_tokens=max_tokens)

        try:
            # Check if managed server is still running
            if self.server_process and self.server_process.poll() is not None:
                print(f"ERROR: LLM server process has died! Exit code: {self.server_process.poll()}")
                raise Exception("LLM server process is not running")

            async with self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=60.0),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"DEBUG: Server returned status {response.status}")
                    print(f"DEBUG: Response: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                return await response.json()
        except aiohttp.ServerDisconnectedError:
            print("DEBUG: Server disconnected during request")
            print(f"DEBUG: Prompt length: {len(prompt)} chars")
            if self.server_process:
                print(f"DEBUG: Server process alive: {self.server_process.poll() is None}")
            raise

    async def score_scholarly_interest(self, passage: str) -> tuple[int, bool]:
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
            result = await self._make_chat_request(prompt, self.BANALITY_SCHEMA, max_tokens=128)
            generated_text = self._extract_text(result)
            if not generated_text:
                return -1, True

            try:
                data = json.loads(generated_text)
                score = int(data["score"])
                if not (1 <= score <= 100):
                    score = -1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                score = -1

            # Score < 40 means banal (not interesting), >= 40 means scholarly/interesting (not banal)
            is_banal = score < 40 if score != -1 else True

            return score, is_banal

        except Exception as e:
            print(f"WARNING: Error scoring passage: {type(e).__name__}: {str(e)}")
            return -1, False

    async def score_scholarly_interest_batch(
        self, passages: list[str], batch_size: int = 4, show_progress: bool = True
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
                temp_alignment = {"target_passage": passage}
                prompt = create_scoring_prompt(temp_alignment)
                payload = self._build_payload(prompt, self.BANALITY_SCHEMA, max_tokens=128)

                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as response:
                    if response.status != 200:
                        return -1, False

                    result = await response.json()
                    generated_text = self._extract_text(result)
                    if not generated_text:
                        return -1, False

                    try:
                        data = json.loads(generated_text)
                        score = int(data["score"])
                        if not (1 <= score <= 100):
                            score = -1
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        score = -1

                    is_banal = score < 40 if score != -1 else True
                    return score, is_banal

            except Exception as e:
                return -1, False

        results = []
        total_passages = len(passages)

        with tqdm(
            total=total_passages,
            desc="LLM Scholarly Evaluation",
            unit="passages",
            disable=not show_progress,
        ) as pbar:
            for i in range(0, total_passages, batch_size):
                batch = passages[i : i + batch_size]

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

    _STANCE_MAP = {1: "Unrelated", 2: "Neutral", 3: "Disagree", 4: "Agree", 5: "Agree"}

    def _parse_llm_response(self, response: str) -> tuple[float, str, str]:
        """Parse structured JSON response from LLM to extract score, reasoning, and stance.

        Returns (score, reasoning, stance) where score is an integer 1-5:
            1 = Unrelated, 2 = Neutral, 3 = Disagree, 4 = Agree (indirect), 5 = Agree (direct)
        """
        try:
            data = json.loads(response)
            reasoning = data.get("reasoning", "No reasoning provided")
            score = int(data["score"])
            stance = self._STANCE_MAP.get(score, "Unknown")
            return float(score), reasoning, stance
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            return 0.0, f"Failed to parse LLM response: {str(e)}", "Unknown"


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

    Respond with a JSON object containing these fields:
    - "score": An integer between 1 and 100
    - "explanation": A brief one-sentence justification

    PASSAGE: "{alignment["target_passage"]}"
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

    prompt = f"""You are an intellectual history expert. Would a scholar cite both passages as evidence of the same specific idea shared between these two authors?

    CRITICAL: Texts from the same intellectual tradition naturally share broad themes (liberty, sovereignty, social contract, religion, etc.). Shared themes are NOT enough. Most thematically related passages should score 2.

    Follow these steps IN ORDER:

    Step 1 — Summarize each passage's core argument in ONE sentence.

    Step 2 — Can you combine both summaries into a SINGLE sentence that accurately captures what both authors argue, without distorting either one?
    - If you cannot: the passages address different questions. Score 1 or 2. STOP here.
    - If you can: write that combined sentence, then proceed to Step 3.

    Step 3 — Test the combined sentence: what is LOST from each passage's argument when you reduce it to this shared claim? State what is lost for each passage.
    - If what is lost is merely stylistic or contextual detail: the core argument is genuinely shared. Proceed to Step 4.
    - If what is lost is the main point of either passage (its specific mechanism, its target, or its conclusion): the combined sentence is too abstract and the passages address different questions. Score 2. STOP here.
    - If what is lost reveals that the authors actually disagree: score 3. STOP here.

    Step 4 — How closely do the arguments align?
    - Same claim but different angles or evidence: score 4.
    - Same claim with overlapping evidence, structure, or framing: score 5.

    Score Guide:
    1 = Unrelated — different subjects entirely.
    2 = Shared domain, different questions — same broad territory but different specific questions. This is the EXPECTED score for most thematically related passages.
    3 = Same question, opposite answer — both engage the same question but reach different conclusions.
    4 = Agree, indirect — same claim from different angles.
    5 = Agree, direct — same claim with overlapping evidence and framing.

    Respond with a JSON object. The "reasoning" field MUST come first:
    - "reasoning": Follow the steps above. For score 4 or 5, you MUST include the combined sentence from Step 2.
    - "score": An integer from 1 to 5

    ---
    Passage 1: {source_text}
    ---
    Passage 2: {target_text}
    ---"""

    return prompt
