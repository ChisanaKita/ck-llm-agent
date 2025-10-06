"""
AgentManager singleton class for CrewAI agent lifecycle management.

This module provides centralized management of CrewAI agents with configuration
from environment variables, async task processing pipeline, and agent lifecycle
management.
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from crewai import Agent, Crew, Task

from ..config import get_settings
from ..models.internal import AgentConfig, AgentTask, TaskStatus
from ..handlers.qwen_vllm import QwenVLLM
from .mcp_registry import MCPRegistry
from .tool_selector import ToolSelector


logger = logging.getLogger(__name__)


class AgentManager:
    """
    Singleton class for managing CrewAI agent instances and task processing.
    
    Provides centralized agent lifecycle management, configuration from environment
    variables, and async task processing pipeline with proper error handling.
    """
    
    _instance: Optional['AgentManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'AgentManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize AgentManager if not already initialized."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.settings = get_settings()
        
        # Agent management
        self._agents: Dict[str, Agent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._default_config: Optional[AgentConfig] = None
        
        # Task processing
        self._active_tasks: Dict[str, AgentTask] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._processing_tasks: bool = False
        self._task_processor_task: Optional[asyncio.Task] = None
        
        # Dependencies
        self._llm_handler: Optional[QwenVLLM] = None
        self._mcp_registry: Optional[MCPRegistry] = None
        self._tool_selector: Optional[ToolSelector] = None
        
        # Performance tracking
        self._total_tasks_processed = 0
        self._total_errors = 0
        self._start_time = time.time()
        
        logger.info("AgentManager singleton initialized")
    
    async def initialize(self) -> None:
        """
        Initialize AgentManager with dependencies and default configuration.
        
        This method should be called once during application startup to set up
        the LLM handler, MCP registry, and default agent configuration.
        """
        try:
            logger.info("Initializing AgentManager dependencies...")
            
            # Initialize LLM handler
            await self._initialize_llm_handler()
            
            # Initialize MCP registry
            await self._initialize_mcp_registry()
            
            # Initialize tool selector
            await self._initialize_tool_selector()
            
            # Create default agent configuration from environment
            self._create_default_config()
            
            # Start task processing pipeline
            await self._start_task_processor()
            
            logger.info("AgentManager initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AgentManager: {e}", exc_info=True)
            raise
    
    async def _initialize_llm_handler(self) -> None:
        """Initialize the QwenVLLM handler."""
        try:
            self._llm_handler = QwenVLLM(
                thinking_mode=True,
                temperature=0.6
            )
            
            # Perform health check
            health_status = await self._llm_handler.check_health()
            if health_status.get("status") != "healthy":
                logger.warning(f"LLM handler health check failed: {health_status}")
            else:
                logger.info("LLM handler initialized and healthy")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM handler: {e}")
            raise
    
    async def _initialize_mcp_registry(self) -> None:
        """Initialize the MCP registry."""
        try:
            self._mcp_registry = MCPRegistry()
            await self._mcp_registry.initialize()
            
            # Get available tools count
            tools = await self._mcp_registry.get_all_tools()
            logger.info(f"MCP registry initialized with {len(tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP registry: {e}")
            # Don't raise - MCP is optional
            self._mcp_registry = None
    
    async def _initialize_tool_selector(self) -> None:
        """Initialize the tool selector."""
        try:
            if self._mcp_registry:
                self._tool_selector = ToolSelector(self._mcp_registry)
                await self._tool_selector.initialize()
                logger.info("Tool selector initialized")
            else:
                logger.warning("Tool selector not initialized - MCP registry unavailable")
                
        except Exception as e:
            logger.error(f"Failed to initialize tool selector: {e}")
            # Don't raise - tool selection is optional
            self._tool_selector = None
    
    def _create_default_config(self) -> None:
        """Create default agent configuration from environment variables."""
        # Get configuration from environment or use defaults
        self._default_config = AgentConfig(
            role=self._get_env_config("AGENT_ROLE", "Intelligent Assistant"),
            goal=self._get_env_config(
                "AGENT_GOAL", 
                "Provide helpful and accurate responses using available tools"
            ),
            backstory=self._get_env_config(
                "AGENT_BACKSTORY",
                "Expert AI assistant with access to various tools and capabilities"
            ),
            verbose=self._get_env_config("AGENT_VERBOSE", True, bool),
            allow_delegation=self._get_env_config("AGENT_ALLOW_DELEGATION", False, bool),
            max_iter=self._get_env_config("AGENT_MAX_ITER", 10, int),
            max_execution_time=self._get_env_config("AGENT_MAX_EXECUTION_TIME", 300, int),
            temperature=self._get_env_config("AGENT_TEMPERATURE", 0.6, float),
            max_tools=self._get_env_config("AGENT_MAX_TOOLS", 5, int),
            tool_selection_strategy=self._get_env_config("AGENT_TOOL_SELECTION", "semantic"),
            memory_enabled=self._get_env_config("AGENT_MEMORY_ENABLED", True, bool),
            context_window=self._get_env_config("AGENT_CONTEXT_WINDOW", 8192, int)
        )
        
        logger.info(f"Default agent configuration created: {self._default_config.role}")
    
    def _get_env_config(self, key: str, default: Any, type_func: type = str) -> Any:
        """Get configuration value from environment with type conversion."""
        import os
        
        value = os.getenv(key, default)
        
        if type_func is bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif type_func in (int, float):
            try:
                return type_func(value)
            except (ValueError, TypeError):
                return default
        else:
            return value
    
    async def _start_task_processor(self) -> None:
        """Start the async task processing pipeline."""
        if not self._processing_tasks:
            self._processing_tasks = True
            self._task_processor_task = asyncio.create_task(self._process_task_queue())
            logger.info("Task processor started")
    
    async def _process_task_queue(self) -> None:
        """Process tasks from the queue asynchronously."""
        logger.info("Task processor pipeline started")
        
        while self._processing_tasks:
            try:
                # Wait for a task with timeout to allow graceful shutdown
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                
                # Process the task
                await self._execute_task(task)
                
                # Mark task as done
                self._task_queue.task_done()
                
            except asyncio.TimeoutError:
                # Normal timeout, continue processing
                continue
            except Exception as e:
                logger.error(f"Error in task processor: {e}", exc_info=True)
                self._total_errors += 1
    
    async def _execute_task(self, task: AgentTask) -> None:
        """
        Execute a single agent task.
        
        Args:
            task: The task to execute
        """
        task_id = task.id
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task_id}")
            
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self._active_tasks[task_id] = task
            
            # Get or create agent for this task
            agent = await self._get_or_create_agent(task.config)
            
            # Select tools for this task
            tools = await self._select_tools_for_task(task)
            
            # Create CrewAI Task
            crew_task = Task(
                description=self._build_task_description(task),
                agent=agent,
                tools=tools,
                expected_output="A helpful and accurate response to the user's request"
            )
            
            # Create and execute crew
            crew = Crew(
                agents=[agent],
                tasks=[crew_task],
                verbose=task.config.verbose
            )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_crew(crew),
                timeout=task.config.max_execution_time
            )
            
            # Process result
            await self._process_task_result(task, result)
            
            # Update statistics
            self._total_tasks_processed += 1
            execution_time = time.time() - start_time
            task.execution_time = execution_time
            
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out")
            task.status = TaskStatus.TIMEOUT
            task.error = f"Task execution timed out after {task.config.max_execution_time}s"
            self._total_errors += 1
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._total_errors += 1
            
        finally:
            # Always update completion time and remove from active tasks
            task.completed_at = datetime.utcnow()
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
    
    async def _execute_crew(self, crew: Crew) -> Any:
        """Execute CrewAI crew in async context."""
        # CrewAI doesn't natively support async, so we run it in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, crew.kickoff)
    
    async def _get_or_create_agent(self, config: AgentConfig) -> Agent:
        """
        Get existing agent or create new one based on configuration.
        
        Args:
            config: Agent configuration
            
        Returns:
            CrewAI Agent instance
        """
        # Create a config key for caching
        config_key = self._get_config_key(config)
        
        if config_key not in self._agents:
            # Create new agent
            agent = Agent(
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                llm=self._llm_handler,
                verbose=config.verbose,
                allow_delegation=config.allow_delegation,
                max_iter=config.max_iter,
                memory=config.memory_enabled
            )
            
            self._agents[config_key] = agent
            self._agent_configs[config_key] = config
            
            logger.info(f"Created new agent: {config.role}")
        
        return self._agents[config_key]
    
    def _get_config_key(self, config: AgentConfig) -> str:
        """Generate a unique key for agent configuration."""
        import hashlib
        
        # Create hash from key configuration parameters
        key_data = f"{config.role}:{config.goal}:{config.backstory}:{config.temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()[:8]
    
    async def _select_tools_for_task(self, task: AgentTask) -> List[Any]:
        """
        Select appropriate tools for the task.
        
        Args:
            task: The task to select tools for
            
        Returns:
            List of tools for the agent
        """
        if not self._tool_selector or not task.tools:
            return []
        
        try:
            # Use semantic selection if available
            if task.config.tool_selection_strategy == "semantic":
                # Build query from messages
                query = self._build_tool_selection_query(task)
                selected_tools = await self._tool_selector.select_tools_semantic(
                    query=query,
                    max_tools=task.config.max_tools
                )
            elif task.config.tool_selection_strategy == "all":
                # Use all available tools (up to limit)
                all_tools = await self._mcp_registry.get_all_tools()
                selected_tools = all_tools[:task.config.max_tools]
            else:
                # Use manually specified tools
                selected_tools = task.tools[:task.config.max_tools]
            
            logger.info(f"Selected {len(selected_tools)} tools for task {task.id}")
            return selected_tools
            
        except Exception as e:
            logger.error(f"Error selecting tools: {e}")
            return []
    
    def _build_tool_selection_query(self, task: AgentTask) -> str:
        """Build a query string for semantic tool selection."""
        # Combine recent messages into a query
        messages = task.messages[-3:]  # Use last 3 messages for context
        query_parts = []
        
        for msg in messages:
            if msg.role == "user":
                query_parts.append(msg.content)
        
        return " ".join(query_parts)
    
    def _build_task_description(self, task: AgentTask) -> str:
        """Build task description from messages."""
        # Use the last user message as the primary task
        for msg in reversed(task.messages):
            if msg.role == "user":
                return msg.content
        
        return "Respond to the user's request"
    
    async def _process_task_result(self, task: AgentTask, result: Any) -> None:
        """
        Process the result from CrewAI execution.
        
        Args:
            task: The original task
            result: Result from CrewAI
        """
        try:
            # Extract content from CrewAI result
            if hasattr(result, 'raw'):
                content = result.raw
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)
            
            # Create result message
            from ..models.api import ChatMessage
            
            task.result = ChatMessage(
                role="assistant",
                content=content
            )
            
            task.status = TaskStatus.COMPLETED
            
            logger.info(f"Task {task.id} result processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing task result: {e}")
            task.status = TaskStatus.FAILED
            task.error = f"Failed to process result: {e}"
    
    async def submit_task(self, task: AgentTask) -> str:
        """
        Submit a task for processing.
        
        Args:
            task: The task to submit
            
        Returns:
            Task ID for tracking
        """
        if not self._processing_tasks:
            raise RuntimeError("AgentManager not initialized or task processor not running")
        
        # Ensure task has an ID
        if not task.id:
            task.id = str(uuid.uuid4())
        
        # Use default config if none provided
        if not task.config:
            task.config = self._default_config
        
        # Add to queue
        await self._task_queue.put(task)
        
        logger.info(f"Task {task.id} submitted for processing")
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[AgentTask]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task object or None if not found
        """
        return self._active_tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was cancelled, False if not found
        """
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            del self._active_tasks[task_id]
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AgentManager performance statistics."""
        uptime = time.time() - self._start_time
        
        return {
            "uptime_seconds": uptime,
            "total_tasks_processed": self._total_tasks_processed,
            "total_errors": self._total_errors,
            "active_tasks": len(self._active_tasks),
            "queue_size": self._task_queue.qsize(),
            "agents_created": len(self._agents),
            "success_rate": (
                (self._total_tasks_processed - self._total_errors) / 
                max(self._total_tasks_processed, 1)
            ),
            "tasks_per_minute": (
                self._total_tasks_processed / max(uptime / 60, 1)
            ),
            "processing_enabled": self._processing_tasks,
            "llm_handler_initialized": self._llm_handler is not None,
            "mcp_registry_initialized": self._mcp_registry is not None,
            "tool_selector_initialized": self._tool_selector is not None
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the AgentManager."""
        logger.info("Shutting down AgentManager...")
        
        # Stop task processing
        self._processing_tasks = False
        
        # Wait for current tasks to complete (with timeout)
        if self._task_processor_task:
            try:
                await asyncio.wait_for(self._task_processor_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Task processor shutdown timed out")
                self._task_processor_task.cancel()
        
        # Cancel remaining active tasks
        for task_id in list(self._active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Shutdown dependencies
        if self._llm_handler:
            await self._llm_handler.connection_pool.close()
        
        if self._mcp_registry:
            await self._mcp_registry.shutdown()
        
        logger.info("AgentManager shutdown completed")
    
    async def process_chat_completion(
        self,
        messages: List[Any],
        model: str,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        thinking_mode: bool = True,
        user: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat completion request through the agent system.
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools for the agent
            tool_choice: Tool choice strategy
            thinking_mode: Enable Qwen3 thinking mode
            user: User identifier
            request_id: Request correlation ID
            
        Returns:
            Dict containing the completion result
        """
        from ..models.internal import AgentTask, AgentConfig
        from ..middleware.metrics import MetricsCollector
        
        start_time = time.time()
        
        try:
            # Create agent configuration
            config = AgentConfig(
                role=self._default_config.role if self._default_config else "Intelligent Assistant",
                goal=self._default_config.goal if self._default_config else "Provide helpful responses",
                backstory=self._default_config.backstory if self._default_config else "Expert AI assistant",
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_mode=thinking_mode,
                verbose=self.settings.debug,
                max_execution_time=self.settings.request_timeout
            )
            
            # Create agent task
            task = AgentTask(
                id=request_id or str(uuid.uuid4()),
                messages=messages,
                tools=tools or [],
                config=config,
                user_id=user,
                model=model
            )
            
            # Submit task and wait for completion
            task_id = await self.submit_task(task)
            
            # Wait for task completion with polling
            max_wait_time = config.max_execution_time
            poll_interval = 0.1
            waited_time = 0
            
            while waited_time < max_wait_time:
                task_status = await self.get_task_status(task_id)
                
                if not task_status:
                    # Task completed and removed from active tasks
                    break
                
                if task_status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
                    break
                
                await asyncio.sleep(poll_interval)
                waited_time += poll_interval
            
            # Get final task result
            final_task = await self.get_task_status(task_id) or task
            
            if final_task.status == TaskStatus.COMPLETED and final_task.result:
                # Extract response components
                content = final_task.result.content
                thinking_content = getattr(final_task.result, 'thinking_content', None)
                tool_calls = getattr(final_task.result, 'tool_calls', [])
                
                # Estimate token usage (in production, this would come from the LLM)
                prompt_tokens = self._estimate_tokens(" ".join([msg.content or "" for msg in messages]))
                completion_tokens = self._estimate_tokens(content or "")
                total_tokens = prompt_tokens + completion_tokens
                
                # Record metrics
                processing_time = time.time() - start_time
                MetricsCollector.record_chat_completion(
                    model=model,
                    status="success",
                    duration=processing_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                return {
                    "content": content,
                    "thinking_content": thinking_content,
                    "tool_calls": tool_calls,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    "processing_time": processing_time
                }
            
            elif final_task.status == TaskStatus.TIMEOUT:
                MetricsCollector.record_chat_completion(model, "timeout", time.time() - start_time)
                raise asyncio.TimeoutError("Request processing timed out")
            
            elif final_task.status == TaskStatus.FAILED:
                MetricsCollector.record_chat_completion(model, "error", time.time() - start_time)
                raise RuntimeError(final_task.error or "Task execution failed")
            
            else:
                MetricsCollector.record_chat_completion(model, "error", time.time() - start_time)
                raise RuntimeError(f"Task completed with unexpected status: {final_task.status}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            MetricsCollector.record_chat_completion(model, "error", processing_time)
            logger.error(f"Chat completion processing failed: {e}", exc_info=True)
            raise
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        This is a rough estimation. In production, you'd use the actual
        tokenizer for the model being used.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimation: ~4 characters per token for English text
        return max(1, len(text) // 4)