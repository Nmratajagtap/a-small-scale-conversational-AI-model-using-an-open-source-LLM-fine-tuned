import pandas as pd

data = {
    'Model': ['LLaMA 2', 'Mistral 7B', 'BLOOM', 'Falcon 7B', 'Gemma 2B'],
    'Size (Parameters)': ['7B, 13B, 70B', '7B', '176B', '7B', '2B, 9B, 27B'],
    'License': ['Meta Llama 2 Community License', 'Apache 2.0', 'Responsible AI License', 'Apache 2.0', 'Gemma Terms of Use'],
    'Fine-tuning Ease': ['Moderate (requires significant resources for larger models)', 'Easy (known for efficient fine-tuning)', 'Difficult (very large model)', 'Moderate', 'Easy'],
    'Performance (General)': ['High', 'High', 'Moderate', 'High', 'High'],
    'Community/Resources': ['Large', 'Large and active', 'Moderate', 'Moderate', 'Growing'],
    'Suitable for Fine-tuning': ['Yes (especially 7B and 13B)', 'Yes', 'Less so due to size', 'Yes', 'Yes']
}

df_llms = pd.DataFrame(data)
print("Research of Open-Source LLMs:")
display(df_llms)

print("\nBased on the research, Mistral 7B appears to be a strong candidate due to its manageable size, permissive license, reported ease of fine-tuning, and active community.")
selected_llm = "Mistral 7B"
