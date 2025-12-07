import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

LLAMA_MODEL_NAME = "models/phi-1_5"

class LlamaGenerator:
    """
    Wraps a LLaMA-style causal LM for CPU-only generation.
    """

    def __init__(
        self,
        model_name: str = LLAMA_MODEL_NAME,
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # CPU → float32
        )
        self.model.to(self.device)

        # Some LLaMA models don't have an explicit pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Build a simple RAG-style prompt.

        We tell the model to answer ONLY from the given Sanskrit context.
        """
        context_block = "\n\n".join(
            [f"[CONTEXT {i+1}]\n{c}" for i, c in enumerate(contexts)]
        )

        prompt = (
            "You are a helpful assistant that answers questions using ONLY the "
            "given Sanskrit context from classical texts.\n"
            "If the answer is not clearly present in the context, say that the "
            "information is not available.\n"
            "You may answer in Sanskrit or simple English explaining the Sanskrit.\n\n"
            f"{context_block}\n\n"
            f"[QUESTION]\n{question}\n\n"
            "[ANSWER]\n"
        )
        return prompt

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        max_context_chunks: int = 3,
    ) -> str:
        """
        Given a user question + retrieved chunks, generate an answer.

        retrieved_chunks: list of dicts with key 'text'
        """
        # Take top-k chunks
        contexts = [c["text"] for c in retrieved_chunks[:max_context_chunks]]
        if not contexts:
            return "सन्दर्भ उपलब्धः नास्ति। I do not have enough context to answer this question."

        prompt = self._build_prompt(question, contexts)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Extract only the part after [ANSWER] (in case model echoes prompt)
        if "[ANSWER]" in full_text:
            answer = full_text.split("[ANSWER]", 1)[-1].strip()
        else:
            # Fallback: remove the prompt prefix
            answer = full_text[len(prompt):].strip()

        return answer


if __name__ == "__main__":
    # Tiny manual sanity test (will be slow the first time, loads the model!)
    gen = LlamaGenerator()
    dummy_contexts = [
        "कालीदासः भोजराजस्य सभायां कविः आसीत् ।",
        "सः अतीव प्राज्ञः कविः आसीत् ।"
    ]
    question = "कालीदासः कस्य राज्ञः सभायां आसीत् ?"
    print(gen.generate_answer(question, [{"text": c} for c in dummy_contexts]))
