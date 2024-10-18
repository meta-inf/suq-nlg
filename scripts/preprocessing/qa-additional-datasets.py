import pickle
from datasets import load_dataset


class Dataset:
    
    def __init__(self):
        pass
    
    def export(self, path):
        lst = []
        for i, rec in enumerate(self.dataset):
            new_rec = self.format_row(rec) | {"id": str(i)}
            lst.append(new_rec)
        with open(path, "wb") as fout:
            pickle.dump(lst, fout)
        return lst


class TruthfulQA(Dataset):
    
    def __init__(self):
        dataset = load_dataset(
            "truthfulqa/truthful_qa", name="generation", split="validation").to_list()
        self.dataset, self.samples = dataset[5:], dataset[:5]

    def format_row(self, row):
        prompt = f"""Answer these questions:"""
        for sample in self.samples:
            prompt += f"\nQ: {sample['question']}\nA: {sample['best_answer']}"
        prompt += f"""\nQ: {row['question']}
A: """
        return {
            "question": row['question'],
            "answer": row['best_answer'],
            "additional_answers": [a for a in row['correct_answers'] if a != row['best_answer']],
            "prompt": prompt
        }

    
class SciQ(Dataset):
    
    def __init__(self):
        self.dataset = load_dataset("allenai/sciq", split="validation")
        self.samples = load_dataset("allenai/sciq", split="train").to_list()[:5]
        
    def format_row(self, row):
        prompt = f"""Answer these questions:"""
        for sample in self.samples:
            prompt += f"\nQ: {sample['question']}\nA: {sample['correct_answer']}"
        prompt += f"""\nQ: {row['question']}
A:"""
        return {
            "question": row['question'],
            "answer": row['correct_answer'],
            "additional_answers": [],
            "prompt": prompt
        }


if __name__ == "__main__":
    TruthfulQA().export("./data/truthful_qa.pkl")
    SciQ().export("./data/sciq.pkl")       