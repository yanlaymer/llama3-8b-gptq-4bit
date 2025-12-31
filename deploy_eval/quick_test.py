"""Quick test script for the quantized model."""

import json
import time
from vllm import LLM, SamplingParams


def main():
    # Load test cases
    with open('medical_test_cases.json') as f:
        data = json.load(f)
    test_cases = data['test_cases'][:5]

    print('Loading nalrunyan/llama3-8b-gptq-4bit model...')
    llm = LLM(
        model='nalrunyan/llama3-8b-gptq-4bit',
        quantization='gptq',
        dtype='half',
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    print('Model loaded successfully!')

    params = SamplingParams(temperature=0.7, max_tokens=256, stop=['<|eot_id|>'])

    # Format prompts
    prompts = []
    for test in test_cases:
        prompt = (
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
            'You are a medical AI assistant.<|eot_id|>'
            '<|start_header_id|>user<|end_header_id|>\n\n'
            f'{test["input"]}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        )
        prompts.append(prompt)

    print(f'\nRunning {len(test_cases)} test cases...')
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    print(f'\n{"="*60}')
    print(f'VALIDATION RESULTS ({elapsed:.1f}s)')
    print(f'{"="*60}\n')

    total_coverage = 0
    for test, output in zip(test_cases, outputs):
        response = output.outputs[0].text
        found = [elem for elem in test['expected_elements'] if elem.lower() in response.lower()]
        coverage = len(found) / len(test['expected_elements']) * 100
        total_coverage += coverage
        status = "PASS" if coverage >= 50 else "WARN"
        print(f'[{status}] {test["id"]}: {test["category"]} - {coverage:.0f}% coverage')

    avg_coverage = total_coverage / len(test_cases)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    print(f'Average Coverage: {avg_coverage:.1f}%')
    print(f'Total Tokens: {total_tokens}')
    print(f'Throughput: {total_tokens/elapsed:.1f} tokens/sec')
    print(f'Latency: {elapsed/len(test_cases):.2f}s per prompt')


if __name__ == '__main__':
    main()
