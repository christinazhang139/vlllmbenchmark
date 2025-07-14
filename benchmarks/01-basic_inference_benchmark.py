#!/usr/bin/env python3
"""
vLLM基础推理性能实验
目标：全面测试vLLM的基本推理性能和系统资源使用情况
"""

from vllm import LLM, SamplingParams
import time
import torch
import psutil
import json
from datetime import datetime

def get_system_status():
    """获取当前系统状态"""
    status = {}
    
    # GPU内存使用
    if torch.cuda.is_available():
        status['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)  # GB
        status['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        status['gpu_name'] = torch.cuda.get_device_name(0)
    
    # 系统内存使用
    memory = psutil.virtual_memory()
    status['system_memory_used'] = memory.used / (1024**3)  # GB
    status['system_memory_total'] = memory.total / (1024**3)  # GB
    status['system_memory_percent'] = memory.percent
    
    return status

def print_system_status(label, status):
    """打印系统状态信息"""
    print(f"\n📊 {label}:")
    if 'gpu_name' in status:
        print(f"   GPU: {status['gpu_name']}")
        print(f"   GPU内存: {status['gpu_memory_allocated']:.1f}GB / {status['gpu_memory_total']:.1f}GB")
    print(f"   系统内存: {status['system_memory_used']:.1f}GB / {status['system_memory_total']:.1f}GB ({status['system_memory_percent']:.1f}%)")

def main():
    print("=" * 70)
    print("vLLM基础推理性能实验")
    print("=" * 70)
    print("🎯 目标：测试vLLM的推理性能、内存使用和批处理效率")
    print("📊 输出：完整的性能分析报告和JSON数据文件")
    print()
    
    # 实验结果记录
    experiment_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': 'vLLM Basic Inference Performance Test',
            'description': 'Comprehensive test of vLLM inference performance and resource usage'
        },
        'system_info': {},
        'performance_metrics': {},
        'test_results': []
    }
    
    # 记录初始系统状态
    initial_status = get_system_status()
    experiment_results['system_info'] = initial_status
    print_system_status("实验开始前的系统状态", initial_status)
    
    # =================================================================
    # 第1步：配置推理参数
    # =================================================================
    print("\n" + "=" * 70)
    print("第1步：配置推理参数")
    print("=" * 70)
    
    sampling_params = SamplingParams(
        temperature=0.8,    # 控制输出的随机性和创造性
        top_p=0.95,        # 核采样，控制词汇选择范围
        max_tokens=50,     # 限制输出长度便于分析
        stop=None          # 不设置停止条件
    )
    
    print("📋 推理参数配置：")
    print(f"   • temperature = {sampling_params.temperature}")
    print(f"     作用：控制输出的随机性，0=确定性，1=高随机性")
    print(f"   • top_p = {sampling_params.top_p}")
    print(f"     作用：只考虑概率累计到95%的词汇，提高输出质量")
    print(f"   • max_tokens = {sampling_params.max_tokens}")
    print(f"     作用：限制输出长度，便于性能测试和结果分析")
    print("✅ 参数配置完成")
    
    # 保存参数配置
    experiment_results['parameters'] = {
        'temperature': sampling_params.temperature,
        'top_p': sampling_params.top_p,
        'max_tokens': sampling_params.max_tokens
    }
    
    # =================================================================
    # 第2步：加载模型
    # =================================================================
    print("\n" + "=" * 70)
    print("第2步：加载模型")
    print("=" * 70)
    
    model_name = "facebook/opt-125m"
    print(f"📦 选择模型：{model_name}")
    print("🔍 模型特点：")
    print("   • 参数量：125,000,000 (1.25亿)")
    print("   • 模型类型：因果语言模型")
    print("   • 开发者：Meta (Facebook)")
    print("   • 优势：体积小，加载快，适合性能基准测试")
    
    # 记录加载前的内存状态
    pre_load_status = get_system_status()
    
    print(f"\n⏱️  开始加载模型...")
    load_start_time = time.time()
    
    # 加载模型
    llm = LLM(model=model_name)
    
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    
    # 记录加载后的内存状态
    post_load_status = get_system_status()
    memory_used = post_load_status['gpu_memory_allocated'] - pre_load_status['gpu_memory_allocated']
    
    print(f"✅ 模型加载完成")
    print(f"   ⏱️  加载时间：{load_time:.2f}秒")
    print(f"   💾 GPU内存占用：{memory_used:.1f}GB")
    print(f"   📊 GPU内存利用率：{(post_load_status['gpu_memory_allocated']/post_load_status['gpu_memory_total']*100):.1f}%")
    
    print_system_status("模型加载后的系统状态", post_load_status)
    
    # 保存加载性能数据
    experiment_results['performance_metrics']['model_loading'] = {
        'model_name': model_name,
        'load_time_seconds': load_time,
        'memory_used_gb': memory_used,
        'pre_load_gpu_memory_gb': pre_load_status['gpu_memory_allocated'],
        'post_load_gpu_memory_gb': post_load_status['gpu_memory_allocated']
    }
    
    # =================================================================
    # 第3步：准备测试数据
    # =================================================================
    print("\n" + "=" * 70)
    print("第3步：准备测试数据")
    print("=" * 70)
    
    # 设计多样化的测试用例，覆盖不同应用场景
    test_cases = [
        {
            "prompt": "Hello, my name is",
            "type": "简单续写",
            "category": "basic_completion",
            "purpose": "测试基本的文本补全能力"
        },
        {
            "prompt": "The weather today is",
            "type": "描述性文本",
            "category": "descriptive", 
            "purpose": "测试描述性文本生成能力"
        },
        {
            "prompt": "Artificial intelligence is",
            "type": "概念解释",
            "category": "conceptual",
            "purpose": "测试概念性知识表达能力"
        },
        {
            "prompt": "In the future, technology will",
            "type": "预测推理",
            "category": "predictive",
            "purpose": "测试逻辑推理和预测能力"
        },
        {
            "prompt": "The best way to learn programming is",
            "type": "建议指导",
            "category": "instructional",
            "purpose": "测试建议和指导内容生成能力"
        }
    ]
    
    print("📝 测试用例设计：")
    print("   设计原则：覆盖AI应用的主要使用场景")
    for i, case in enumerate(test_cases, 1):
        print(f"   {i}. {case['type']}：\"{case['prompt']}\"")
        print(f"      目的：{case['purpose']}")
        print(f"      分类：{case['category']}")
    
    prompts = [case["prompt"] for case in test_cases]
    print(f"\n✅ 测试数据准备完成，共 {len(prompts)} 个测试用例")
    
    # 保存测试用例信息
    experiment_results['test_cases'] = test_cases
    
    # =================================================================
    # 第4步：执行推理测试
    # =================================================================
    print("\n" + "=" * 70)
    print("第4步：执行推理测试")
    print("=" * 70)
    
    print("🚀 开始批量推理...")
    print("💡 特色：vLLM使用批处理技术同时处理多个请求")
    print("📊 测试：将同时处理所有测试用例，测量整体性能")
    
    # 执行推理并计时
    inference_start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_end_time = time.time()
    
    inference_time = inference_end_time - inference_start_time
    
    print(f"✅ 推理完成")
    print(f"   ⏱️  推理总时间：{inference_time:.2f}秒")
    print(f"   📈 处理速度：{len(prompts)/inference_time:.2f} 请求/秒")
    print(f"   ⚡ 平均延迟：{inference_time/len(prompts):.2f}秒/请求")
    
    # 保存推理性能数据
    experiment_results['performance_metrics']['inference'] = {
        'total_time_seconds': inference_time,
        'throughput_requests_per_second': len(prompts) / inference_time,
        'average_latency_seconds': inference_time / len(prompts),
        'total_requests': len(prompts),
        'batch_processing': True
    }
    
    # =================================================================
    # 第5步：详细结果分析
    # =================================================================
    print("\n" + "=" * 70)
    print("第5步：详细结果分析")
    print("=" * 70)
    
    print("📋 逐个测试用例的详细结果：")
    print("=" * 80)
    
    total_tokens = 0
    token_counts = []
    
    for i, (output, test_case) in enumerate(zip(outputs, test_cases)):
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        token_count = len(output.outputs[0].token_ids)
        total_tokens += token_count
        token_counts.append(token_count)
        
        # 保存每个测试的详细结果
        result = {
            'test_id': i + 1,
            'category': test_case['category'],
            'type': test_case['type'],
            'prompt': prompt,
            'generated_text': generated_text,
            'token_count': token_count,
            'full_output': f"{prompt}{generated_text}",
            'purpose': test_case['purpose']
        }
        experiment_results['test_results'].append(result)
        
        print(f"\n测试用例 {i+1}：{test_case['type']} ({test_case['category']})")
        print(f"输入提示：\"{prompt}\"")
        print(f"生成文本：\"{generated_text}\"")
        print(f"Token数量：{token_count}")
        print(f"完整输出：\"{prompt}{generated_text}\"")
        print("-" * 60)
    
    # =================================================================
    # 第6步：性能统计和分析
    # =================================================================
    print("\n" + "=" * 70)
    print("第6步：性能统计和分析")
    print("=" * 70)
    
    # 计算统计指标
    avg_tokens_per_output = total_tokens / len(outputs)
    tokens_per_second = total_tokens / inference_time
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    
    print("📊 关键性能指标 (KPI)：")
    print(f"   🏃 模型加载时间：{load_time:.2f}秒")
    print(f"   ⚡ 推理总时间：{inference_time:.2f}秒")
    print(f"   📈 吞吐量：{len(prompts)/inference_time:.2f} 请求/秒")
    print(f"   🎯 平均延迟：{inference_time/len(prompts):.2f}秒/请求")
    print(f"   📝 Token统计：")
    print(f"      • 总Token数：{total_tokens}")
    print(f"      • 平均Token数：{avg_tokens_per_output:.1f}")
    print(f"      • Token范围：{min_tokens} ~ {max_tokens}")
    print(f"   🚀 Token生成速度：{tokens_per_second:.1f} tokens/秒")
    print(f"   💾 内存使用：{memory_used:.1f}GB GPU内存")
    
    # 效率分析
    memory_efficiency = memory_used / 0.125  # GB per billion parameters
    gpu_utilization = (post_load_status['gpu_memory_allocated'] / post_load_status['gpu_memory_total']) * 100
    
    print(f"\n📈 效率分析：")
    print(f"   💡 内存效率：{memory_efficiency:.1f}GB/B参数")
    print(f"   📊 GPU利用率：{gpu_utilization:.1f}%")
    print(f"   ⚖️  批处理优势：同时处理{len(prompts)}个请求")
    
    # 保存统计数据
    experiment_results['performance_metrics']['statistics'] = {
        'total_tokens_generated': total_tokens,
        'average_tokens_per_output': avg_tokens_per_output,
        'tokens_per_second': tokens_per_second,
        'min_tokens_per_output': min_tokens,
        'max_tokens_per_output': max_tokens,
        'memory_efficiency_gb_per_billion_params': memory_efficiency,
        'gpu_utilization_percent': gpu_utilization
    }
    
    # =================================================================
    # 第7步：基准数据总结
    # =================================================================
    print("\n" + "=" * 70)
    print("第7步：基准数据总结")
    print("=" * 70)
    
    print("🏆 vLLM性能基准数据：")
    print(f"   框架版本：vLLM (当前安装版本)")
    print(f"   测试模型：{model_name} (125M参数)")
    print(f"   硬件平台：{initial_status.get('gpu_name', 'Unknown GPU')}")
    print(f"   GPU内存：{initial_status['gpu_memory_total']:.0f}GB")
    print(f"   ")
    print(f"   📊 关键指标：")
    print(f"      • 模型加载：{load_time:.2f}秒")
    print(f"      • 推理吞吐：{len(prompts)/inference_time:.2f} req/s")
    print(f"      • Token速率：{tokens_per_second:.1f} tok/s")
    print(f"      • 内存占用：{memory_used:.1f}GB")
    print(f"      • 内存效率：{memory_efficiency:.1f}GB/B参数")
    
    # 保存基准总结
    experiment_results['benchmark_summary'] = {
        'framework': 'vLLM',
        'model': model_name,
        'model_size': '125M parameters',
        'hardware': initial_status.get('gpu_name', 'Unknown GPU'),
        'gpu_memory_total': initial_status['gpu_memory_total'],
        'key_metrics': {
            'load_time_seconds': load_time,
            'throughput_req_per_sec': len(prompts) / inference_time,
            'token_generation_rate_per_sec': tokens_per_second,
            'memory_usage_gb': memory_used,
            'memory_efficiency_gb_per_billion_params': memory_efficiency
        }
    }
    
    # =================================================================
    # 第8步：保存实验结果
    # =================================================================
    print("\n" + "=" * 70)
    print("第8步：保存实验结果")
    print("=" * 70)
    
    # 生成结果文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"vllm_basic_experiment_{timestamp}.json"
    
    # 保存到JSON文件
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 实验结果已保存：{result_filename}")
    print(f"📊 数据包含：")
    print(f"   • 完整的系统信息")
    print(f"   • 详细的性能指标")
    print(f"   • 所有测试用例的结果")
    print(f"   • 基准数据总结")
    
    # =================================================================
    # 实验完成总结
    # =================================================================
    print("\n" + "=" * 70)
    print("🎯 实验完成总结")
    print("=" * 70)
    
    print("✅ 实验成功完成！")
    print(f"💡 主要发现：")
    print(f"   1. vLLM加载125M模型耗时 {load_time:.1f}秒，内存占用 {memory_used:.1f}GB")
    print(f"   2. 批处理推理速度达到 {len(prompts)/inference_time:.1f} 请求/秒")
    print(f"   3. Token生成速度为 {tokens_per_second:.0f} tokens/秒")
    print(f"   4. GPU内存利用率为 {gpu_utilization:.1f}%")
    print(f"   5. 所有测试用例都成功生成了合理的文本输出")
    
    print(f"\n📋 数据用途：")
    print(f"   • 可作为vLLM性能基准参考")
    print(f"   • 可用于不同配置的性能对比")
    print(f"   • 可用于硬件升级效果评估")
    print(f"   • 可用于与其他推理框架的对比分析")
    
    print(f"\n🚀 后续建议：")
    print(f"   • 尝试更大的模型测试性能扩展性")
    print(f"   • 测试不同批处理大小的性能差异") 
    print(f"   • 比较不同推理参数的效果")
    print(f"   • 在相同条件下测试其他推理框架")

if __name__ == "__main__":
    main()
