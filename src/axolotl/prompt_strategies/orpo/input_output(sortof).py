from typing import Any, Dict, Optional

from pydantic import BaseModel

from axolotl.prompt_tokenizers import IGNORE_INDEX, PromptTokenizingStrategy
from axolotl.prompters import Prompter

def load(
    tokenizer,
    cfg,
    ds_cfg: Optional[Dict[str, Any]] = None, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    return ORPOTokenizingStrategy(
        "placeholder",  # Assuming a placeholder for whatever identifier or setting is needed
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len
    )

class ORPOTokenizingStrategy(PromptTokenizingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_prompt(self, prompt):
        base_input_ids = []
        base_labels = []

        # Prepare base inputs and labels from setup or instructions
        for segment in prompt["segments"]:
            text = segment.get("text", "")  # Safeguard against missing 'text'
            if not segment["label"]:  # These are part of setup or instructions and should be ignored
                if text:  # Ensure there is text to encode
                    part_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    base_input_ids += part_input_ids
                    base_labels += [IGNORE_INDEX] * len(part_input_ids)

        # Extract and tokenize accepted and rejected outputs
        chosen_input_ids, rejected_input_ids = base_input_ids[:], base_input_ids[:]
        chosen_labels, rejected_labels = base_labels[:], base_labels[:]

        for segment in prompt["segments"]:
            if segment["label"]:  # Train on these parts
                # Process accepted output
                accept_text = segment.get("text_accept", "")
                if accept_text:  # Check if there's text to process
                    accept_input_ids = self.tokenizer.encode(accept_text, add_special_tokens=False)
                    chosen_input_ids += accept_input_ids
                    chosen_labels += accept_input_ids  # Labels are the same as input_ids

                # Process rejected output
                reject_text = segment.get("text_reject", "")
                if reject_text:  # Check if there's text to process
                    reject_input_ids = self.tokenizer.encode(reject_text, add_special_tokens=False)
                    rejected_input_ids += reject_input_ids
                    rejected_labels += reject_input_ids  # Labels are the same as input_ids for rejected part

        return {
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": [1] * len(rejected_labels),
            "input_ids": chosen_input_ids,
            "labels": chosen_labels,
            "attention_mask": [1] * len(chosen_labels),
            "prompt_attention_mask": (
                [1] * len(base_input_ids) + [0] * (len(chosen_labels) - len(base_input_ids))
            )
        }

