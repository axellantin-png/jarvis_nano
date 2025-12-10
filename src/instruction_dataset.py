import json
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                instr = obj["instruction"]
                resp = obj["response"]
                self.examples.append((instr, resp))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        instruction, response = self.examples[idx]

        user_prefix = "<|user|> "
        assistant_prefix = "<|assistant|> "

        # On encode séparément pour savoir où est la frontière
        user_text = user_prefix + instruction
        assistant_text = assistant_prefix + response

        user_ids = self.tokenizer.encode(
            user_text,
            add_special_tokens=False
        )
        assistant_ids = self.tokenizer.encode(
            assistant_text,
            add_special_tokens=False
        )

        input_ids = user_ids + assistant_ids
        # labels: ignore côté user, apprend côté assistant
        labels = [-100] * len(user_ids) + assistant_ids

        # Troncature
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # Padding
        pad_id = self.tokenizer.pad_token_id
        attention_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_length:
            input_ids.append(pad_id)
            labels.append(-100)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }