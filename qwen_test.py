from speculative_prefill import enable_prefill_spec
import os
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

# monkey patch must be placed before everything
enable_prefill_spec(
    spec_model='Qwen/Qwen2.5-7B-Instruct', 
    spec_config_path='./configs/config_p1_full_lah8.yaml'
)

from vllm import LLM, SamplingParams

if __name__ == '__main__':
    llm = LLM(
        'Qwen/QwQ-32B',
        gpu_memory_utilization=0.8, 
        enforce_eager=True, 
        enable_chunked_prefill=False, 
        tensor_parallel_size=2
    )


    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    outputs = llm.generate(prompts, sampling_params)
    
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print("LLM instance created successfully.")
