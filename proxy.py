from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import time
import uuid
import asyncio
import json
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSIFICATION_SERVER_URL = "http://localhost:8001/classify"

app = FastAPI(
    title="Smart Batching Classification Proxy",
    description="Smart proxy with intelligent batching for sub-4 second performance"
)

class ProxyRequest(BaseModel):
    """Request model for single text classification"""
    sequence: str

class ProxyResponse(BaseModel):
    """Response model containing classification result ('code' or 'not code')"""
    result: str

class SmartBatchingProxy:
    def __init__(self):
        self.http_client = None
        self.pending_requests = {}  # request_id -> (sequence, future)
        self.batch_size = 5
        self.batch_timeout = 0.0001  # 0.1ms to collect batch - extremely aggressive
        self.server_semaphore = asyncio.Semaphore(1)
        self.processing_task = None
        
    async def initialize(self):
        """Initialize HTTP client and start background processing"""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=1.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
            http2=True
        )
        
        # Start background processing task
        self.processing_task = asyncio.create_task(self._background_processor())
        logger.info("SmartBatchingProxy initialized")
    
    async def classify(self, sequence: str) -> str:
        """Classify a single sequence with smart batching"""
        request_id = str(uuid.uuid4())
        
        # Create a future for this request
        future = asyncio.Future()
        self.pending_requests[request_id] = (sequence, future)
        
        # Wait for result with very short timeout
        try:
            result = await asyncio.wait_for(future, timeout=2.0)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out")
            return "timeout"
        finally:
            # Clean up
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
    
    async def _background_processor(self):
        """Background task that continuously processes batches"""
        while True:
            try:
                if len(self.pending_requests) >= self.batch_size:
                    # Process immediately if we have a full batch
                    await self._process_batch()
                elif len(self.pending_requests) > 0:
                    # Wait extremely briefly for more requests
                    await asyncio.sleep(self.batch_timeout)
                    if len(self.pending_requests) > 0:
                        await self._process_batch()
                else:
                    # No requests, sleep very briefly
                    await asyncio.sleep(0.0001)
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                await asyncio.sleep(0.001)
    
    async def _process_batch(self):
        """Process current batch of requests"""
        if not self.pending_requests:
            return
        
        # Take up to batch_size requests
        batch_items = list(self.pending_requests.items())[:self.batch_size]
        sequences = [item[1][0] for item in batch_items]
        request_ids = [item[0] for item in batch_items]
        
        # Remove from pending
        for request_id in request_ids:
            del self.pending_requests[request_id]
        
        # Process batch
        async with self.server_semaphore:
            try:
                response = await self.http_client.post(
                    CLASSIFICATION_SERVER_URL,
                    json={"sequences": sequences},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data["results"]
                    
                    # Set results for all requests in batch
                    for request_id, result in zip(request_ids, results):
                        # Find the original future and set result
                        for req_id, (seq, future) in list(self.pending_requests.items()):
                            if seq == sequences[request_ids.index(request_id)]:
                                if not future.done():
                                    future.set_result(result)
                                break
                        else:
                            # If not found in pending, it might be in the batch_items
                            for req_id, (seq, future) in batch_items:
                                if seq == sequences[request_ids.index(request_id)]:
                                    if not future.done():
                                        future.set_result(result)
                                    break
                    
                    logger.info(f"🚀 Smart batch of {len(sequences)} requests processed")
                    
                elif response.status_code == 429:
                    # Rate limited - process requests individually
                    logger.warning("Rate limited, processing requests individually")
                    for request_id, (sequence, future) in batch_items:
                        if not future.done():
                            individual_result = await self._process_single(sequence)
                            future.set_result(individual_result)
                else:
                    # Error - set error for all requests
                    for request_id, (sequence, future) in batch_items:
                        if not future.done():
                            future.set_result("error")
                            
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Set error for all requests
                for request_id, (sequence, future) in batch_items:
                    if not future.done():
                        future.set_result("error")
    
    async def _process_single(self, sequence: str) -> str:
        """Process a single request"""
        try:
            response = await self.http_client.post(
                CLASSIFICATION_SERVER_URL,
                json={"sequences": [sequence]},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["results"][0]
            else:
                return "error"
        except Exception as e:
            logger.error(f"Single request error: {e}")
            return "error"
    
    async def shutdown(self):
        """Shutdown the proxy"""
        if self.processing_task:
            self.processing_task.cancel()
        if self.http_client:
            await self.http_client.aclose()

# Global proxy instance
proxy = SmartBatchingProxy()

@app.on_event("startup")
async def startup_event():
    await proxy.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await proxy.shutdown()

@app.post("/proxy_classify")
async def proxy_classify(req: ProxyRequest):
    """Proxy endpoint for classification requests"""
    result = await proxy.classify(req.sequence)
    return ProxyResponse(result=result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pending_requests": len(proxy.pending_requests)
    }
