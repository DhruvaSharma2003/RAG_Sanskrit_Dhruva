import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


QWEN_MODEL_PATH = "models/qwen1.5b"


class PhiGenerator:   
    """
    Wraps a Qwen2.5–1.5B Instruct causal LM for CPU-only generation.
    """

    def __init__(
        self,
        model_name: str = QWEN_MODEL_PATH,
        device: str = "cpu",
        max_new_tokens: int = 128,       
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # ----------------------------------------------------------
        # Load Model + Tokenizer
        # ----------------------------------------------------------
        print(f"✓ Loading Qwen model from: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)

     
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print("✓ Qwen2.5–1.5B-Instruct loaded successfully (CPU mode).")

    # -----------------------------------------------------------
    # Build Prompt
    # -----------------------------------------------------------
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Build a RAG-style prompt for Sanskrit context-based answering.
        Qwen follows instructions much better than Phi-1.5.
        """

        context_block = "\n\n".join(
            [f"[CONTEXT {i+1}]\n{c}" for i, c in enumerate(contexts)]
        )

        prompt = (
            "You are a helpful assistant that answers questions ONLY from the "
            "given Sanskrit context. If the answer is not present, respond with: "
            "'Information not available in the given Sanskrit text.'\n"
            "Keep the answer short and precise.\n\n"
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
        max_context_chunks: int = 1,   
    ) -> str:

        if not retrieved_chunks:
            return "सन्दर्भः उपलब्धः नास्ति । (No relevant context found.)"

        
        contexts = [c["text"] for c in retrieved_chunks[:max_context_chunks]]
        prompt = self._build_prompt(question, contexts)

        # Tokenization
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generation
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode to text
        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Output extraction
        if "[ANSWER]" in full_text:
            answer = full_text.split("[ANSWER]", 1)[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()

        return answer


# -----------------------------------------------------------
# Manual Test (For Debugging)
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
