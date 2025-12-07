import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local Phi-1.5 model path
PHI_MODEL_PATH = "models/phi-1_5"


class PhiGenerator:
    """
    Wraps a Phi-1.5 causal LM for CPU-only generation.
    """

    def __init__(
        self,
        model_name: str = PHI_MODEL_PATH,
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
            torch_dtype=torch.float32,
        ).to(self.device)

        # Fix pad tokens for Phi
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print("✓ Phi-1.5 loaded successfully (CPU mode).")

    # -----------------------------------------------------------
    # Prompt Builder
    # -----------------------------------------------------------
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Build a RAG-style prompt for Sanskrit context-based answering.
        """
        context_block = "\n\n".join(
            [f"[CONTEXT {i+1}]\n{c}" for i, c in enumerate(contexts)]
        )

        prompt = (
            "You are a helpful assistant that answers questions using ONLY the "
            "given Sanskrit context from classical texts.\n"
            "If the answer is not present in the context, clearly say: "
            "'Information not available in the provided Sanskrit text.'\n"
            "You may respond in simple Sanskrit or English.\n\n"
            f"{context_block}\n\n"
            f"[QUESTION]\n{question}\n\n"
            "[ANSWER]\n"
        )
        return prompt

    # -----------------------------------------------------------
    # Answer Generation
    # -----------------------------------------------------------
    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        max_context_chunks: int = 3,
    ) -> str:

        if not retrieved_chunks:
            return "सन्दर्भः उपलब्धः नास्ति । (No relevant context found.)"

        # Extract context texts
        contexts = [c["text"] for c in retrieved_chunks[:max_context_chunks]]
        prompt = self._build_prompt(question, contexts)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Extract answer
        if "[ANSWER]" in full_text:
            answer = full_text.split("[ANSWER]", 1)[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()

        return answer


# -----------------------------------------------------------
# Manual Test
# -----------------------------------------------------------
if __name__ == "__main__":
    gen = PhiGenerator()
    dummy_contexts = [
        "कालीदासः भोजराजस्य सभायां कविः आसीत् ।",
        "सः अतीव प्राज्ञः कविः आसीत् ।"
    ]
    question = "कालीदासः कस्य राज्ञः सभायां आसीत् ?"

    print("\nGenerated Answer:\n")
    print(gen.generate_answer(question, [{"text": c} for c in dummy_contexts]))
