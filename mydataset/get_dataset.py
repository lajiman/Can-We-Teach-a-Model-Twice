from datasets import load_dataset

def get_dataset(name):
    assert name == "de" or name == "es", 'name should be de for German or es for Spanish'
    dataset = load_dataset("mlsum", name=name)
    if name == "de":
        return dataset.map(summary_head_de)
    return dataset.map(summary_head_es)
    

def summary_head_de(examples): 
    head = "zusammenfassen: "
    examples["text"] = head + examples["text"]
    return examples

def summary_head_es(examples): 
    head = "resumir: "
    examples["text"] = head + examples["text"]
    return examples

def show_samples(dataset, num_samples=5, seed=73):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Text: {example['text']}'")
        print(f"'>> Summary: {example['summary']}'")


if __name__ == "__main__":
    dataset = get_dataset("es")
    print(dataset)
    # show_samples(dataset, num_samples=10)