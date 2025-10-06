"""
Utility for testing vLLM endpoint configurations.

This module provides command-line utilities and functions for testing
vLLM endpoint connectivity, configuration validation, and health monitoring.
"""

import asyncio
import logging
import sys
from typing import Optional, Dict, Any
import argparse

from ..config import VLLMEndpointConfig
from ..services.config_validation import get_config_service
from ..services.health_service import get_health_service
# Avoid circular import - import endpoint_manager when needed


logger = logging.getLogger(__name__)


class EndpointTester:
    """
    Utility class for testing vLLM endpoint configurations.
    
    Provides methods for testing connectivity, validating configurations,
    and monitoring endpoint health.
    """
    
    def __init__(self):
        """Initialize the endpoint tester."""
        self.config_service = get_config_service()
        self.health_service = get_health_service()
        self.endpoint_manager = get_endpoint_manager()
    
    async def test_endpoint(
        self, 
        base_url: str, 
        model: str = "Qwen/Qwen3-8B-AWQ",
        api_key: Optional[str] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Test a single endpoint configuration.
        
        Args:
            base_url: Endpoint URL
            model: Model name
            api_key: Optional API key
            timeout: Request timeout
            
        Returns:
            Test results
        """
        print(f"Testing endpoint: {base_url}")
        print(f"Model: {model}")
        print("-" * 50)
        
        # Create test configuration
        config = VLLMEndpointConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout
        )
        
        # Validate configuration
        validation_result = await self.config_service.validator.validate_endpoint_config(config)
        
        # Print validation results
        print("Configuration Validation:")
        print(f"  Valid: {validation_result['valid']}")
        print(f"  Endpoint Type: {validation_result['endpoint_type'].value if validation_result['endpoint_type'] else 'Unknown'}")
        
        if validation_result['issues']:
            print("  Issues:")
            for issue in validation_result['issues']:
                print(f"    - {issue}")
        
        if validation_result['warnings']:
            print("  Warnings:")
            for warning in validation_result['warnings']:
                print(f"    - {warning}")
        
        if validation_result['recommendations']:
            print("  Recommendations:")
            for rec in validation_result['recommendations']:
                print(f"    - {rec}")
        
        # Print connectivity results
        connectivity = validation_result.get('connectivity', {})
        print("\nConnectivity Test:")
        print(f"  Connected: {connectivity.get('connected', False)}")
        if connectivity.get('response_time_ms'):
            print(f"  Response Time: {connectivity['response_time_ms']:.2f}ms")
        if connectivity.get('error'):
            print(f"  Error: {connectivity['error']}")
        
        # Print model information
        model_info = validation_result.get('model_info')
        if model_info and 'data' in model_info:
            print("\nAvailable Models:")
            for model_data in model_info['data'][:5]:  # Show first 5 models
                print(f"  - {model_data.get('id', 'Unknown')}")
            if len(model_info['data']) > 5:
                print(f"  ... and {len(model_info['data']) - 5} more")
        
        return validation_result
    
    async def test_chat_completion(
        self,
        base_url: str,
        model: str = "Qwen/Qwen3-8B-AWQ", 
        api_key: Optional[str] = None,
        message: str = "Hello, how are you?"
    ) -> Dict[str, Any]:
        """
        Test chat completion functionality.
        
        Args:
            base_url: Endpoint URL
            model: Model name
            api_key: Optional API key
            message: Test message
            
        Returns:
            Chat completion test results
        """
        print(f"Testing chat completion: {base_url}")
        print(f"Test message: {message}")
        print("-" * 50)
        
        try:
            # Create temporary endpoint
            from ..handlers.qwen_vllm import QwenVLLM
            
            handler = QwenVLLM.create_for_external(
                base_url=base_url,
                model=model,
                api_key=api_key
            )
            
            # Test chat completion
            messages = [{"role": "user", "content": message}]
            
            start_time = asyncio.get_event_loop().time()
            response = await handler.call(messages)
            end_time = asyncio.get_event_loop().time()
            
            response_time = (end_time - start_time) * 1000
            
            result = {
                "success": True,
                "response_time_ms": response_time,
                "response": response,
                "error": None
            }
            
            print("Chat Completion Test:")
            print("  Success: True")
            print(f"  Response Time: {response_time:.2f}ms")
            print(f"  Content Length: {len(response.get('content', ''))}")
            
            if response.get('thinking_content'):
                print(f"  Thinking Content Length: {len(response['thinking_content'])}")
            
            if response.get('usage'):
                usage = response['usage']
                print("  Token Usage:")
                print(f"    Prompt: {usage.get('prompt_tokens', 0)}")
                print(f"    Completion: {usage.get('completion_tokens', 0)}")
                print(f"    Total: {usage.get('total_tokens', 0)}")
            
            # Close handler
            await handler.close()
            
        except Exception as e:
            result = {
                "success": False,
                "response_time_ms": None,
                "response": None,
                "error": str(e)
            }
            
            print("Chat Completion Test:")
            print("  Success: False")
            print(f"  Error: {str(e)}")
        
        return result
    
    async def test_all_configured_endpoints(self) -> Dict[str, Any]:
        """
        Test all configured endpoints.
        
        Returns:
            Test results for all endpoints
        """
        print("Testing all configured endpoints...")
        print("=" * 60)
        
        results = {}
        
        # Test configuration validation
        validation_results = await self.config_service.validate_all_endpoints()
        
        for endpoint_name, validation in validation_results.items():
            print(f"\n{endpoint_name.upper()} ENDPOINT:")
            print("-" * 30)
            
            print(f"Valid: {validation['valid']}")
            print(f"Type: {validation['endpoint_type'].value if validation['endpoint_type'] else 'Unknown'}")
            
            connectivity = validation.get('connectivity', {})
            print(f"Connected: {connectivity.get('connected', False)}")
            
            if connectivity.get('response_time_ms'):
                print(f"Response Time: {connectivity['response_time_ms']:.2f}ms")
            
            if validation['issues']:
                print("Issues:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
            
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
            
            results[endpoint_name] = validation
        
        # Test system health
        print("\nSYSTEM HEALTH:")
        print("-" * 30)
        
        health_response = await self.health_service.check_system_health()
        print(f"Overall Status: {health_response.status.value}")
        print(f"Uptime: {health_response.uptime_seconds:.1f}s")
        
        if health_response.services:
            print("Service Health:")
            for service_name, service_health in health_response.services.items():
                print(f"  {service_name}: {service_health.status.value}")
                if service_health.message:
                    print(f"    {service_health.message}")
        
        results["system_health"] = health_response
        
        return results
    
    async def benchmark_endpoint(
        self,
        base_url: str,
        model: str = "Qwen/Qwen3-8B-AWQ",
        api_key: Optional[str] = None,
        num_requests: int = 10,
        concurrency: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark endpoint performance.
        
        Args:
            base_url: Endpoint URL
            model: Model name
            api_key: Optional API key
            num_requests: Number of requests to make
            concurrency: Number of concurrent requests
            
        Returns:
            Benchmark results
        """
        print(f"Benchmarking endpoint: {base_url}")
        print(f"Requests: {num_requests}, Concurrency: {concurrency}")
        print("-" * 50)
        
        from ..handlers.qwen_vllm import QwenVLLM
        
        # Create handler
        handler = QwenVLLM.create_for_external(
            base_url=base_url,
            model=model,
            api_key=api_key
        )
        
        # Test messages
        test_messages = [
            [{"role": "user", "content": "What is the capital of France?"}],
            [{"role": "user", "content": "Explain quantum computing in simple terms."}],
            [{"role": "user", "content": "Write a short poem about technology."}],
            [{"role": "user", "content": "What are the benefits of renewable energy?"}],
            [{"role": "user", "content": "How does machine learning work?"}],
        ]
        
        async def make_request(request_id: int) -> Dict[str, Any]:
            """Make a single request."""
            messages = test_messages[request_id % len(test_messages)]
            
            try:
                start_time = asyncio.get_event_loop().time()
                response = await handler.call(messages)
                end_time = asyncio.get_event_loop().time()
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "response_time": (end_time - start_time) * 1000,
                    "tokens": response.get('usage', {}).get('total_tokens', 0),
                    "error": None
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "response_time": None,
                    "tokens": 0,
                    "error": str(e)
                }
        
        # Run benchmark
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(request_id: int):
            async with semaphore:
                return await make_request(request_id)
        
        start_time = asyncio.get_event_loop().time()
        
        tasks = [limited_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            tokens = [r['tokens'] for r in successful_requests]
            
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            total_tokens = sum(tokens)
            
            throughput = len(successful_requests) / total_time
            tokens_per_second = total_tokens / total_time
        else:
            avg_response_time = min_response_time = max_response_time = 0
            total_tokens = throughput = tokens_per_second = 0
        
        benchmark_result = {
            "total_requests": num_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "total_time_seconds": total_time,
            "avg_response_time_ms": avg_response_time,
            "min_response_time_ms": min_response_time,
            "max_response_time_ms": max_response_time,
            "throughput_rps": throughput,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "errors": [r['error'] for r in failed_requests if r['error']]
        }
        
        # Print results
        print("Benchmark Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Successful: {len(successful_requests)}")
        print(f"  Failed: {len(failed_requests)}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        
        if successful_requests:
            print(f"  Avg Response Time: {avg_response_time:.2f}ms")
            print(f"  Min Response Time: {min_response_time:.2f}ms")
            print(f"  Max Response Time: {max_response_time:.2f}ms")
            print(f"  Total Tokens: {total_tokens}")
            print(f"  Tokens/Second: {tokens_per_second:.2f}")
        
        if failed_requests:
            print("  Errors:")
            for error in set(benchmark_result['errors']):
                print(f"    - {error}")
        
        # Close handler
        await handler.close()
        
        return benchmark_result


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test vLLM endpoint configurations")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test endpoint command
    test_parser = subparsers.add_parser('test', help='Test a specific endpoint')
    test_parser.add_argument('url', help='Endpoint URL')
    test_parser.add_argument('--model', default='Qwen/Qwen3-8B-AWQ', help='Model name')
    test_parser.add_argument('--api-key', help='API key')
    test_parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds')
    
    # Test chat command
    chat_parser = subparsers.add_parser('chat', help='Test chat completion')
    chat_parser.add_argument('url', help='Endpoint URL')
    chat_parser.add_argument('--model', default='Qwen/Qwen3-8B-AWQ', help='Model name')
    chat_parser.add_argument('--api-key', help='API key')
    chat_parser.add_argument('--message', default='Hello, how are you?', help='Test message')
    
    # Test all command
    subparsers.add_parser('test-all', help='Test all configured endpoints')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark endpoint performance')
    bench_parser.add_argument('url', help='Endpoint URL')
    bench_parser.add_argument('--model', default='Qwen/Qwen3-8B-AWQ', help='Model name')
    bench_parser.add_argument('--api-key', help='API key')
    bench_parser.add_argument('--requests', type=int, default=10, help='Number of requests')
    bench_parser.add_argument('--concurrency', type=int, default=3, help='Concurrent requests')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    tester = EndpointTester()
    
    try:
        if args.command == 'test':
            await tester.test_endpoint(args.url, args.model, args.api_key, args.timeout)
        
        elif args.command == 'chat':
            await tester.test_chat_completion(args.url, args.model, args.api_key, args.message)
        
        elif args.command == 'test-all':
            await tester.test_all_configured_endpoints()
        
        elif args.command == 'benchmark':
            await tester.benchmark_endpoint(
                args.url, args.model, args.api_key, args.requests, args.concurrency
            )
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())