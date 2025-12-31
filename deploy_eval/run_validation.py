"""Run full validation and save JSON results."""

import json
import time
from datetime import datetime
from vllm import LLM, SamplingParams


def main():
    # Load test cases
    with open('medical_test_cases.json') as f:
        data = json.load(f)
    test_cases = data['test_cases']

    print('Loading nalrunyan/llama3-8b-gptq-4bit model...')
    llm = LLM(
        model='nalrunyan/llama3-8b-gptq-4bit',
        quantization='gptq',
        dtype='half',
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    print('Model loaded successfully!')

    params = SamplingParams(temperature=0.7, max_tokens=512, stop=['<|eot_id|>'])

    # Format prompts
    prompts = []
    for test in test_cases:
        prompt = (
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
            'You are a medical AI assistant. Provide accurate, evidence-based medical information. '
            'Be thorough and professional.<|eot_id|>'
            '<|start_header_id|>user<|end_header_id|>\n\n'
            f'{test["input"]}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        )
        prompts.append(prompt)

    print(f'\nRunning {len(test_cases)} test cases...')
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    # Process results
    detailed_results = []
    category_scores = {}

    for test, output in zip(test_cases, outputs):
        response = output.outputs[0].text
        tokens_generated = len(output.outputs[0].token_ids)

        # Check expected elements
        found = [elem for elem in test['expected_elements'] if elem.lower() in response.lower()]
        missing = [elem for elem in test['expected_elements'] if elem.lower() not in response.lower()]
        coverage = len(found) / len(test['expected_elements']) * 100 if test['expected_elements'] else 100

        result = {
            'test_id': test['id'],
            'category': test['category'],
            'task': test['task'],
            'difficulty': test['difficulty'],
            'input': test['input'],
            'expected_elements': test['expected_elements'],
            'generated_output': response,
            'elements_found': found,
            'elements_missing': missing,
            'coverage_score': round(coverage, 1),
            'tokens_generated': tokens_generated,
            'passed': coverage >= 60
        }
        detailed_results.append(result)

        # Aggregate by category
        cat = test['category']
        if cat not in category_scores:
            category_scores[cat] = {'total': 0, 'sum': 0, 'passed': 0, 'count': 0}
        category_scores[cat]['sum'] += coverage
        category_scores[cat]['count'] += 1
        category_scores[cat]['passed'] += 1 if coverage >= 60 else 0

    # Calculate final metrics
    total_tokens = sum(r['tokens_generated'] for r in detailed_results)
    avg_coverage = sum(r['coverage_score'] for r in detailed_results) / len(detailed_results)
    passed_count = sum(1 for r in detailed_results if r['passed'])

    category_summary = {}
    for cat, data in category_scores.items():
        category_summary[cat] = {
            'test_count': data['count'],
            'avg_coverage': round(data['sum'] / data['count'], 1),
            'pass_rate': round(data['passed'] / data['count'] * 100, 1)
        }

    # Build final JSON
    validation_results = {
        'metadata': {
            'model': 'nalrunyan/llama3-8b-gptq-4bit',
            'date': datetime.now().isoformat(),
            'hardware': 'GCP g2-standard-4 (NVIDIA L4 23GB)',
            'engine': 'vLLM v0.13.0',
            'quantization': 'GPTQ 4-bit'
        },
        'summary': {
            'total_tests': len(detailed_results),
            'passed_tests': passed_count,
            'pass_rate': round(passed_count / len(detailed_results) * 100, 1),
            'average_coverage': round(avg_coverage, 1),
            'total_tokens_generated': total_tokens,
            'total_time_seconds': round(elapsed, 2),
            'throughput_tokens_per_sec': round(total_tokens / elapsed, 1),
            'avg_latency_per_prompt_sec': round(elapsed / len(detailed_results), 2)
        },
        'category_breakdown': category_summary,
        'detailed_results': detailed_results
    }

    # Save to JSON
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)

    print(f'\n{"="*60}')
    print('VALIDATION COMPLETE')
    print(f'{"="*60}')
    print(f'Total Tests: {len(detailed_results)}')
    print(f'Passed: {passed_count} ({passed_count/len(detailed_results)*100:.1f}%)')
    print(f'Average Coverage: {avg_coverage:.1f}%')
    print(f'Throughput: {total_tokens/elapsed:.1f} tokens/sec')
    print(f'\nResults saved to: validation_results.json')


if __name__ == '__main__':
    main()
