# References

Based on info from: https://www.youtube.com/watch?v=r-cbg5ON60Q&t=1s

Models: https://huggingface.co/unsloth/phi-4-GGUF/tree/main

Model being tried: https://huggingface.co/unsloth/phi-4-GGUF/blob/main/phi-4-Q5_K_M.gguf

wget -O phi-4-Q5_K_M.gguf "https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q5_K_M.gguf"



Found templates with system prompts for different LLMs:
https://github.com/maryasov/ollama-models-instruct-for-cline/tree/main
The idea cam from here: https://huggingface.co/benhaotang/Rombos-Coder-V2.5-Qwen-7b-GGUF_cline



# Experiences

## olilanz-qwen-v2.5 based on MHKetbi/Qwen2.5-Coder-32B-Instruct-Roo:q8_0 (32bn / 32k)

Seems to have best results, but is very slow. Asks sensible questions.

## rombos-coder-v2.5 based on hf.co/bartowski/Rombos-Coder-V2.5-Qwen-7b-GGUF:F16 (7bn / 96k)

Optimal VRAM.

ERROR: Roo Code uses complex prompts and iterative task execution that may be challenging for less capable models. For best results, it's recommended to use Claude 3.5 Sonnet for its advanced agentic coding capabilities.

## olilanz-phi4 based on hf.co/unsloth/phi-4-GGUF:Q5_K_M (14bn / 56k)

Optimal VRAM.

Starts off well, but then eds in error.

ERROR: Unexpected API Response: The language model did not provide any assistant messages. This may indicate an issue with the API or the model's output

## olilanz-deepseek-coder from hf.co/tensorblock/deepseek-coder-6.7b-base-GGUF:Q8_0 (6.7bn / 32k)

ERROR: Roo Code uses complex prompts and iterative task execution that may be challenging for less capable models. For best results, it's recommended to use Claude 3.5 Sonnet for its advanced agentic coding capabilities.

## olilanz-deepseek-r1 from  tom_himanen/deepseek-r1-roo-cline-tools:8b (8bn / 96k)

Roo Code uses complex prompts and iterative task execution that may be challenging for less capable models. For best results, it's recommended to use Claude 3.5 Sonnet for its advanced agentic coding capabilities.

## olilanz-codellama from  codellama:13b-code (13bn / 56k)

endless loop
