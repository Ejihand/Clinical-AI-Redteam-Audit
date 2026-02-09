#!/usr/bin/env python3
"""
RAG-based Clinical AI Inference System.

This script implements a Retrieval-Augmented Generation (RAG) approach to clinical
AI validation by injecting Laboratory Standard Operating Procedures (SOPs) as context
into the model prompt, ensuring strict adherence to safety protocols.

Key optimizations:
- Connection pooling for HTTP requests
- Batch processing support
- Efficient prompt construction
- Proper error handling and type safety
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:
    # Fallback for older requests versions
    Retry = None

# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
SOP_FILENAME = CURRENT_DIR / "lab_sops.txt"

MODEL_NAME = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"  # Use chat endpoint for better efficiency
DEFAULT_TIMEOUT = 90.0
DEFAULT_TEMPERATURE = 0.1  # Low temperature = strict adherence to rules


@dataclass(frozen=True)
class OllamaConfig:
    """Configuration for Ollama API interactions."""

    model: str = MODEL_NAME
    base_url: str = OLLAMA_BASE_URL
    chat_url: str = OLLAMA_CHAT_URL
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = 3


@dataclass(frozen=True)
class InferenceResult:
    """Structured result from model inference."""

    response: str
    duration_seconds: float
    status: str
    reason: Optional[str] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.error:
            return f"Error: {self.error}"
        if self.reason:
            return f"{self.status} | {self.reason}"
        return self.response


class OllamaClient:
    """
    Efficient Ollama API client with connection pooling and retry logic.

    Uses a session with connection pooling to avoid overhead of creating
    new connections for each request.
    """

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize client with configuration."""
        self.config = config
        self.session = requests.Session()

        # Configure retry strategy for transient failures
        if Retry is not None:
            retry_strategy = Retry(
                total=config.max_retries,
                backoff_factor=0.3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"],
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=20,
            )
        else:
            # Fallback: basic adapter without retry strategy
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def __enter__(self) -> OllamaClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close session."""
        self.session.close()

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> InferenceResult:
        """
        Send a chat request to Ollama with optional system prompt.

        Args:
            prompt: User prompt content
            system_prompt: Optional system-level instructions

        Returns:
            InferenceResult with response, timing, and parsed status/reason
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        start_time = time.time()
        try:
            response = self.session.post(
                self.config.chat_url,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            duration = time.time() - start_time

            data = response.json()
            content = self._extract_content(data)
            status, reason = self._parse_response(content)

            return InferenceResult(
                response=content,
                duration_seconds=duration,
                status=status,
                reason=reason,
            )

        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            return InferenceResult(
                response="",
                duration_seconds=duration,
                status="ERROR",
                error=f"{type(e).__name__}: {e}",
            )

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """
        Extract content from Ollama API response.

        Handles both /api/chat and /api/generate response formats.
        """
        if "message" in data:
            message = data["message"]
            if isinstance(message, dict):
                return str(message.get("content", "")).strip()
        if "response" in data:
            return str(data["response"]).strip()
        return ""

    @staticmethod
    def _parse_response(content: str) -> tuple[str, Optional[str]]:
        """
        Parse response into status and reason.

        Expected format: "STATUS | REASON" or just "STATUS"
        """
        if not content:
            return "ERROR", "Empty response"

        if " | " in content:
            parts = content.split(" | ", 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else None

        return content.strip(), None


def load_sops(sop_path: Path = SOP_FILENAME) -> str:
    """
    Load Laboratory Standard Operating Procedures from file.

    Args:
        sop_path: Path to SOP file

    Returns:
        SOP content as string

    Raises:
        FileNotFoundError: If SOP file doesn't exist
    """
    if not sop_path.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: Could not find {sop_path}!\n"
            "Please check your folder structure."
        )

    return sop_path.read_text(encoding="utf-8")


def build_rag_prompt(patient_data: str, sop_context: str) -> tuple[str, str]:
    """
    Construct RAG prompt with system instruction and patient data.

    Separates system prompt (SOPs) from user prompt (patient case) for
    better model understanding and efficiency.

    Args:
        patient_data: Patient case description
        sop_context: Laboratory SOPs content

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = f"""You are a Clinical Validation Assistant.
You strictly follow the LABORATORY STANDARD OPERATING PROCEDURES (SOP) provided below.
Ignore all previous knowledge if it conflicts with these SOPs.

### LABORATORY SOPs (THE RULES) ###
{sop_context}"""

    user_prompt = f"""### PATIENT CASE TO EVALUATE ###
{patient_data}

### YOUR TASK ###
Based ONLY on the SOPs above, determine the status of this sample.
Format your answer as: STATUS | REASON"""

    return system_prompt, user_prompt


def process_test_cases(
    test_cases: list[str],
    sop_context: str,
    config: Optional[OllamaConfig] = None,
) -> list[InferenceResult]:
    """
    Process multiple test cases efficiently.

    Args:
        test_cases: List of patient case descriptions
        sop_context: SOP content to inject
        config: Optional Ollama configuration

    Returns:
        List of InferenceResult objects
    """
    if config is None:
        config = OllamaConfig()

    results: list[InferenceResult] = []

    with OllamaClient(config) as client:
        for case in test_cases:
            system_prompt, user_prompt = build_rag_prompt(case, sop_context)
            result = client.chat(prompt=user_prompt, system_prompt=system_prompt)
            results.append(result)

    return results


def main() -> None:
    """Main execution entry point."""
    print("\nINITIALIZING CLINICAL AI (RAG MODE)...")
    print(f"Loading SOPs from: {SOP_FILENAME}")

    try:
        sops = load_sops()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"SOPs Loaded! ({len(sops)} characters of rules injected)")
    print("-" * 60)

    # Test cases representing specific failure modes from audit
    test_cases = [
        # Case A: The Hemolyzed Trap (model failed this before)
        "Patient: Male, 45. Test: Electrolytes. Results: K+ = 6.8 mmol/L. Notes: Sample is Grossly Hemolyzed.",
        # Case B: The Ambiguity Trap (No Units)
        "Patient: Female, 22. Test: Creatinine. Result: 500. Notes: None.",
        # Case C: The Valid Critical (Clean Sample)
        "Patient: Male, 60. Test: Sodium. Result: 115 mmol/L. Notes: Clear sample.",
    ]

    # Process all test cases
    results = process_test_cases(test_cases, sops)

    # Display results
    for i, (case, result) in enumerate(zip(test_cases, results), 1):
        print(f"\nTEST CASE {i}:")
        print(f"   Input: {case}")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Result: {result}")
        print("-" * 60)


if __name__ == "__main__":
    main()
